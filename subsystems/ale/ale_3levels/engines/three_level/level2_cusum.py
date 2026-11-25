"""L2 CUSUM over standardized forecast residuals, with frozen calibration.

L2 accumulates the same signed residual that L1 scores instantaneously:

    z[t] = (observed[t] - forecast[t]) / sigma
    e[t] = -z[t] for a negative instability direction, +z[t] otherwise
    C_instability[t] = max(0, C_instability[t-1] + e[t] - k)
    C_overforecast[t] = max(0, C_overforecast[t-1] - e[t] - k)

Both sums start at zero on the first monitoring acquisition and are never
reset. An L2 alarm is ``C_instability > h``. ``C_overforecast`` is exported as
an informational diagnostic (the forecast persistently predicts more
deformation than observed) and never raises an alarm. The residual mean is not
subtracted; calibration bias is reported instead, so a biased model surfaces.

Sigma is the sample standard deviation of the *aggregated* residual series on
pre-monitoring dates. The pooled per-pixel L1 sigma is only used for the
per-pixel L2 map, where the same per-pixel residuals are accumulated.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from ale_3levels.engines.three_level.calibration import (
    SCHEMA_VERSION,
    check_identity,
    compatible_schema,
)
from ale_3levels.engines.three_level.level1_zscore import (
    positive_finite,
    residual_scale,
)
from ale_3levels.spatial import SpatialCoherenceRegion

LOGGER = logging.getLogger(__name__)

FORMULA = (
    'z = (observed - forecast) / sigma; e = direction * z; '
    'C_instability = max(0, C_instability + e - k); '
    'C_overforecast = max(0, C_overforecast - e - k); alarm = C_instability > h'
)
METHOD = 'aggregate_residual_sample_std_ddof1_no_demeaning'


@dataclass(frozen=True)
class Level2Parameters:
    """Operational L2 constants shared by every L2 series."""

    reference_value: float
    decision_threshold: float
    direction: str
    bias_warning_ratio: float = 0.25

    def __post_init__(self) -> None:
        if not np.isfinite(self.reference_value) or self.reference_value < 0:
            raise ValueError('level2.reference_value must be finite and non-negative.')
        positive_finite(self.decision_threshold, 'level2.decision_threshold')
        positive_finite(self.bias_warning_ratio, 'level2.bias_warning_ratio')
        if self.direction not in {'negative', 'positive'}:
            raise ValueError(
                'L2 direction must be negative or positive; set level2.direction '
                'when anomaly_detection.direction is both.'
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Level2Parameters:
        """Read ``level2`` with direction defaulting to the L1 direction."""
        settings = config.get('level2', {})
        if not isinstance(settings, dict):
            raise ValueError('level2 must be a mapping.')  # noqa: TRY004 (prototype)
        direction = settings.get(
            'direction', config.get('anomaly_detection', {}).get('direction')
        )
        return cls(
            reference_value=float(settings.get('reference_value', 0.5)),
            decision_threshold=float(settings.get('decision_threshold', 5.0)),
            direction=str(direction).lower(),
            bias_warning_ratio=float(settings.get('bias_warning_ratio', 0.25)),
        )

    @property
    def sign(self) -> float:
        """Multiplier that makes instability-direction residuals positive."""
        return -1.0 if self.direction == 'negative' else 1.0


@dataclass(frozen=True)
class Level2Result:
    """Per-acquisition L2 series for the TSF and coherence-qualified regions.

    ``z`` covers calibration and monitoring dates. CUSUM values before
    monitoring (or before a region's activation) are NaN.
    """

    z: np.ndarray
    instability_cusum: np.ndarray
    overforecast_cusum: np.ndarray
    alarm: np.ndarray
    first_alarm_index: int | None
    regional_available: np.ndarray
    regional_instability_cusum: np.ndarray
    regional_overforecast_cusum: np.ndarray
    regional_alarm: np.ndarray
    regional_first_alarm_index: int | None
    regional_sigmas: tuple[dict[str, Any], ...]
    parameters: Level2Parameters
    calibration: dict[str, Any]


def residual_cusum(
    z: np.ndarray,
    reference_value: float,
    direction: str,
    start_index: int,
    resets: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal instability and over-forecast CUSUM.

    Missing standardized residuals hold the previous statistic. Both sums
    restart from zero at every index in ``resets`` (B2.6; the prototype
    never resets).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError('L2 CUSUM input must be one-dimensional.')
    if not 0 <= start_index <= z.size:
        raise ValueError('L2 CUSUM start index is outside the series.')
    sign = -1.0 if direction == 'negative' else 1.0
    instability = np.full(z.size, np.nan, dtype=np.float64)
    overforecast = np.full(z.size, np.nan, dtype=np.float64)
    upper = lower = 0.0
    for index in range(start_index, z.size):
        if index in resets:
            upper = lower = 0.0
        value = z[index]
        if np.isfinite(value):
            evidence = sign * value
            upper = max(0.0, upper + evidence - reference_value)
            lower = max(0.0, lower - evidence - reference_value)
        instability[index], overforecast[index] = upper, lower
    return instability, overforecast


def level2_alarms(
    instability: np.ndarray, decision_threshold: float
) -> tuple[np.ndarray, int | None]:
    """Return strict ``C > h`` alarm flags and the first alarm index."""
    alarm = np.isfinite(instability) & (instability > decision_threshold)
    indices = np.flatnonzero(alarm)
    return alarm, (int(indices[0]) if indices.size else None)


def aggregate_residual(residual_stack: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Mean residual over a fixed support, using finite pixels per date."""
    valid = np.broadcast_to(support, residual_stack.shape) & np.isfinite(residual_stack)
    count = np.sum(valid, axis=(1, 2))
    return np.divide(
        np.sum(np.where(valid, residual_stack, 0.0), axis=(1, 2)),
        count,
        out=np.full(residual_stack.shape[0], np.nan, dtype=np.float64),
        where=count > 0,
    )


def calibration_diagnostics(values: np.ndarray, sigma: float) -> dict[str, Any]:
    """Bias and autocorrelation of a calibration residual series."""
    finite = values[np.isfinite(values)]
    mean = float(np.mean(finite))
    pairs = np.column_stack((values[:-1], values[1:]))
    pairs = pairs[np.all(np.isfinite(pairs), axis=1)]
    lag1 = (
        float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
        if pairs.shape[0] >= 3 and np.all(np.std(pairs, axis=0) > 0)
        else None
    )
    return {
        'mean_residual': mean,
        'bias_ratio': mean / sigma,
        'lag1_autocorrelation': lag1,
    }


def calibration_replay(
    z_calibration: np.ndarray, parameters: Level2Parameters
) -> dict[str, int]:
    """Count alarm episodes when L2 is replayed over in-sample calibration z."""
    instability, _ = residual_cusum(
        z_calibration, parameters.reference_value, parameters.direction, 0
    )
    alarm, _ = level2_alarms(instability, parameters.decision_threshold)
    episodes = int(np.sum(alarm[1:] & ~alarm[:-1]) + (alarm[0] if alarm.size else 0))
    return {
        'samples': int(np.sum(np.isfinite(z_calibration))),
        'alarm_episodes': episodes,
    }


def frozen_level2_calibration(
    path: Path,
    residual_stack: np.ndarray,
    dates: tuple[str, ...],
    calibration_window: tuple[int, int],
    monitoring_start: int,
    mask: np.ndarray,
    *,
    unit: str,
    target_feature: str,
    parameters: Level2Parameters,
    population: str = 'prototype',
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the TSF-wide aggregate sigma once, before monitoring; reject changes.

    Mirrors the L1 artifact contract: exclusive creation, pre-monitoring
    cutoff, unit/target checks and a fingerprint of the calibration history.
    With ``population='calibration_split'`` (B2.2) the caller has set records
    outside the independent calibration split to NaN: the aggregate series
    then exists only on calibration-split dates.
    """
    method = METHOD if population == 'prototype' else f'{METHOD}+{population}'
    if residual_stack.shape != (len(dates), *mask.shape):
        raise ValueError(
            'L2 residual stack, dates and mask must share a grid/time axis.'
        )
    parsed = tuple(date.fromisoformat(value) for value in dates)
    if any(right <= left for left, right in pairwise(parsed)):
        raise ValueError('L2 acquisition dates must be unique and chronological.')
    if not 0 <= monitoring_start < len(dates):
        raise ValueError('L2 monitoring start is outside acquisition dates.')
    start, end = calibration_window
    if not 0 <= start < end <= len(dates):
        raise ValueError('L2 calibration window is outside acquisition dates.')
    end = min(end, monitoring_start)
    series = aggregate_residual(residual_stack, mask)[start:end]
    has_history = start < end and np.any(np.isfinite(series))
    fingerprint = None
    if has_history:
        digest = hashlib.sha256()
        digest.update(json.dumps(dates[start:end]).encode())
        digest.update(str(mask.shape).encode())
        digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
        digest.update(np.asarray(series, dtype='<f8').tobytes())
        fingerprint = digest.hexdigest()
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        check_identity(payload, identity, 'L2')
        if (
            not compatible_schema(payload.get('schema_version'))
            or payload.get('method') != method
            or payload.get('unit') != unit
            or payload.get('target_feature') != target_feature
        ):
            raise ValueError(
                'Frozen L2 calibration has an incompatible method, unit or target.'
            )
        positive_finite(payload['sigma'], 'Frozen L2 calibration sigma')
        if (
            not payload.get('last_date')
            or payload['last_date'] >= dates[monitoring_start]
        ):
            raise ValueError('Frozen L2 calibration must end before monitoring starts.')
        if fingerprint is not None and payload.get('input_fingerprint') != fingerprint:
            raise ValueError(
                'L2 calibration history/support changed. Use a new calibration_file '
                'or run directory for an explicitly recalibrated model.'
            )
    else:
        if not has_history:
            raise ValueError(
                'No finite L2 calibration residuals precede monitoring. Configure a '
                'prior calibration period or supply an existing calibration_file.'
            )
        sigma, count = residual_scale(series)
        used = list(dates[start:end])
        if population != 'prototype':
            used = [
                day
                for day, value in zip(dates[start:end], series)
                if np.isfinite(value)
            ]
        payload = {
            'schema_version': SCHEMA_VERSION,
            'method': method,
            'formula': FORMULA,
            'population': (
                'daily_mean_residual_on_static_tsf_mask'
                if population == 'prototype'
                else f'daily_mean_residual_on_static_tsf_mask,{population}_records'
            ),
            'unit': unit,
            'target_feature': target_feature,
            'sigma': sigma,
            'sample_count': count,
            **calibration_diagnostics(series, sigma),
            'first_date': used[0],
            'last_date': used[-1],
            'calibration_dates': used,
            'input_fingerprint': fingerprint,
        }
        if identity is not None:
            payload['identity'] = dict(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents silently replacing another run's calibration.
        with path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
    warn_on_bias(payload, parameters, 'TSF-wide')
    return payload


def warn_on_bias(
    calibration: dict[str, Any], parameters: Level2Parameters, label: str
) -> bool:
    """Log a calibration bias warning; return whether the limit is exceeded."""
    ratio = float(calibration['bias_ratio'])
    exceeded = abs(ratio) > parameters.bias_warning_ratio
    if exceeded:
        LOGGER.warning(
            '%s L2 calibration bias |mean/sigma| = %.2f exceeds %.2f; the forecast '
            'is systematically offset and L2 detection timing is affected.',
            label,
            abs(ratio),
            parameters.bias_warning_ratio,
        )
    return exceeded


def compute_level2(
    residual_stack: np.ndarray,
    mask: np.ndarray,
    calibration_window: tuple[int, int],
    monitoring_window: tuple[int, int],
    calibration: dict[str, Any],
    parameters: Level2Parameters,
    regions: tuple[SpatialCoherenceRegion, ...] = (),
    calibration_dates: np.ndarray | None = None,
    center: float = 0.0,
    center_regions: bool = False,
    resets: frozenset[int] = frozenset(),
) -> Level2Result:
    """Calculate TSF-wide and coherence-qualified regional L2 series.

    Each region is standardized by the sigma of its own aggregate residual on
    pre-monitoring dates, over the support frozen at activation. Its CUSUM
    starts at the activation acquisition; earlier monitoring residuals helped
    select the region and would bias its statistic upwards. With
    ``calibration_dates`` (B2.2) only those dates fit the regional sigma.
    """
    time_count = residual_stack.shape[0]
    monitoring_start, monitoring_end = monitoring_window
    if not 0 <= monitoring_start < monitoring_end <= time_count:
        raise ValueError('L2 monitoring window is outside the acquisition range.')
    sigma = positive_finite(calibration['sigma'], 'L2 sigma')
    # B2.1: ``center`` is the frozen calibration mean (0 = prototype).
    z = (aggregate_residual(residual_stack, mask) - center) / sigma
    z[monitoring_end:] = np.nan
    instability, overforecast = residual_cusum(
        z, parameters.reference_value, parameters.direction, monitoring_start, resets
    )
    instability[monitoring_end:] = np.nan
    overforecast[monitoring_end:] = np.nan
    alarm, first_alarm = level2_alarms(instability, parameters.decision_threshold)

    regional_instability = np.full(time_count, np.nan, dtype=np.float64)
    regional_overforecast = np.full(time_count, np.nan, dtype=np.float64)
    regional_available = np.zeros(time_count, dtype=bool)
    regional_sigmas: list[dict[str, Any]] = []
    calibration_end = min(calibration_window[1], monitoring_start)
    for region in regions:
        series = aggregate_residual(residual_stack, region.support)
        baseline = series[calibration_window[0] : calibration_end]
        if calibration_dates is not None:
            selected = calibration_dates[calibration_window[0] : calibration_end]
            baseline = np.where(selected, baseline, np.nan)
        try:
            region_sigma, count = residual_scale(baseline)
        except ValueError:
            LOGGER.warning(
                'Skipping L2 for the coherent region activated at index %d: its '
                'fixed support has insufficient pre-monitoring residuals.',
                region.activation_index,
            )
            continue
        diagnostics = calibration_diagnostics(baseline, region_sigma)
        region_center = diagnostics['mean_residual'] if center_regions else 0.0
        warn_on_bias(diagnostics, parameters, f'Region@{region.activation_index}')
        regional_sigmas.append(
            {
                'activation_index': int(region.activation_index),
                'support_pixels': int(np.sum(region.support)),
                'sigma': region_sigma,
                'sample_count': count,
                **diagnostics,
            }
        )
        start = max(region.activation_index, monitoring_start)
        if start >= monitoring_end:
            continue
        positive, negative = residual_cusum(
            (series - region_center) / region_sigma,
            parameters.reference_value,
            parameters.direction,
            start,
            resets,
        )
        active = np.zeros(time_count, dtype=bool)
        active[start:monitoring_end] = True
        regional_available |= active
        regional_instability = np.where(
            active, np.fmax(regional_instability, positive), regional_instability
        )
        regional_overforecast = np.where(
            active, np.fmax(regional_overforecast, negative), regional_overforecast
        )
    regional_alarm, regional_first_alarm = level2_alarms(
        regional_instability, parameters.decision_threshold
    )
    return Level2Result(
        z=z,
        instability_cusum=instability,
        overforecast_cusum=overforecast,
        alarm=alarm,
        first_alarm_index=first_alarm,
        regional_available=regional_available,
        regional_instability_cusum=regional_instability,
        regional_overforecast_cusum=regional_overforecast,
        regional_alarm=regional_alarm,
        regional_first_alarm_index=regional_first_alarm,
        regional_sigmas=tuple(regional_sigmas),
        parameters=parameters,
        calibration=calibration,
    )


def pixel_level2(
    residual_stack: np.ndarray,
    sigma: float,
    mask: np.ndarray,
    monitoring_window: tuple[int, int],
    parameters: Level2Parameters,
    center: float = 0.0,
    resets: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel L2: CUSUM of L1 z, using the frozen per-pixel L1 sigma.

    Returns the maximum monitoring-period instability CUSUM and the first
    alarm acquisition index (NaN where no alarm or outside ``mask``).
    """
    sigma = positive_finite(sigma, 'L1 sigma')
    start, end = monitoring_window
    upper = np.zeros(mask.shape, dtype=np.float64)
    maximum = np.zeros(mask.shape, dtype=np.float64)
    first_alarm = np.full(mask.shape, np.nan, dtype=np.float64)
    observed = np.zeros(mask.shape, dtype=bool)
    for index in range(start, end):
        if index in resets:
            upper = np.zeros(mask.shape, dtype=np.float64)
        evidence = parameters.sign * (residual_stack[index] - center) / sigma
        finite = np.isfinite(evidence)
        observed |= finite
        upper = np.where(
            finite,
            np.maximum(0.0, upper + evidence - parameters.reference_value),
            upper,
        )
        maximum = np.maximum(maximum, upper)
        newly = np.isnan(first_alarm) & (upper > parameters.decision_threshold)
        first_alarm[newly] = index
    valid = mask & observed
    return np.where(valid, maximum, np.nan), np.where(valid, first_alarm, np.nan)


def pixel_level2_series(
    residual_stack: np.ndarray,
    sigma: float,
    mask: np.ndarray,
    monitoring_window: tuple[int, int],
    parameters: Level2Parameters,
    center: float = 0.0,
    resets: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-date per-pixel instability and over-forecast CUSUM (ALE addition).

    The same recursion as :func:`pixel_level2` (which keeps only the maximum
    and the first alarm), with the over-forecast side as in
    :func:`residual_cusum`. NaN outside the monitoring window and ``mask``.
    """
    sigma = positive_finite(sigma, 'L1 sigma')
    start, end = monitoring_window
    upper = np.zeros(mask.shape, dtype=np.float64)
    lower = np.zeros(mask.shape, dtype=np.float64)
    instability = np.full(residual_stack.shape, np.nan, dtype=np.float64)
    overforecast = np.full(residual_stack.shape, np.nan, dtype=np.float64)
    for index in range(start, end):
        if index in resets:
            upper = np.zeros(mask.shape, dtype=np.float64)
            lower = np.zeros(mask.shape, dtype=np.float64)
        evidence = parameters.sign * (residual_stack[index] - center) / sigma
        finite = np.isfinite(evidence)
        upper = np.where(
            finite,
            np.maximum(0.0, upper + evidence - parameters.reference_value),
            upper,
        )
        lower = np.where(
            finite,
            np.maximum(0.0, lower - evidence - parameters.reference_value),
            lower,
        )
        instability[index] = np.where(mask, upper, np.nan)
        overforecast[index] = np.where(mask, lower, np.nan)
    return instability, overforecast


def level2_metadata(result: Level2Result, dates: tuple[str, ...]) -> dict[str, Any]:
    """JSON-ready L2 contract block for monitoring metadata and exports."""
    parameters = result.parameters
    first = result.first_alarm_index
    regional_first = result.regional_first_alarm_index
    # A calibration reused from an earlier run may have no dates in this stack.
    calibration_dates = set(result.calibration.get('calibration_dates', []))
    selected = np.array([value in calibration_dates for value in dates], dtype=bool)
    return {
        'formula': FORMULA,
        'direction': parameters.direction,
        'reference_value': parameters.reference_value,
        'decision_threshold': parameters.decision_threshold,
        'comparison': 'strict',
        'reset_after_alarm': False,
        'alarm_side': 'instability',
        'informational_side': 'overforecast',
        'bias_correction': 'none',
        'bias_warning_ratio': parameters.bias_warning_ratio,
        'bias_warning': abs(float(result.calibration['bias_ratio']))
        > parameters.bias_warning_ratio,
        'calibration': result.calibration,
        'calibration_replay': (
            calibration_replay(result.z[selected], parameters)
            if np.any(selected)
            else None
        ),
        'first_alarm_date': None if first is None else dates[first],
        'regional_first_alarm_date': (
            None if regional_first is None else dates[regional_first]
        ),
        'regional_calibration': list(result.regional_sigmas),
    }
