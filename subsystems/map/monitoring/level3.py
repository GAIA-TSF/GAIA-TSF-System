"""L3 regime-change evidence: a heuristic score plus an explicit evidence rule.

The regime-change score is a dimensionless heuristic in [0, 1). It is *not* a
probability of regime change or failure:

    u[t] = direction * (EMA(d observed / dt) - EMA(d forecast / dt))
    z[t] = (u[t] - mean_calibration) / sigma_calibration
    score[t] = EMA((1 - exp(-max(0, z[t]))) * sign_persistence(z)[t])

The mean and sigma (sample std, ddof=1) are fitted on pre-monitoring dates only.
Without smoothing and at full sign persistence, a score threshold ``s``
corresponds to ``z = -ln(1 - s)``.

L3 levels per monitoring acquisition, using the larger of the TSF-wide and
coherent-region scores:

    watch: score >= watch_threshold for N consecutive acquisitions,
           or an L2 alarm (TSF-wide or regional) is active
    alert: L2 alarm active AND score >= alert_threshold for N consecutive
           acquisitions AND a coherence-qualified region exists

The score alone never raises an alert.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.monitoring.level1 import positive_finite
from subsystems.map.monitoring.level2 import Level2Result
from subsystems.map.monitoring.spatial_coherence import SpatialCoherenceRegion
from subsystems.map.utils.contract import SCHEMA_VERSION, compatible_schema

LOGGER = logging.getLogger(__name__)

SCORE_FORMULA = (
    'u = direction * (EMA(d observed/dt) - EMA(d forecast/dt)); '
    'z = (u - mean_calibration) / sigma_calibration; '
    'score = EMA((1 - exp(-max(0, z))) * sign_persistence(z))'
)
EVIDENCE_RULE = {
    'watch': 'score >= watch_threshold for N consecutive acquisitions OR L2 alarm',
    'alert': (
        'L2 alarm AND score >= alert_threshold for N consecutive acquisitions '
        'AND coherence-qualified region exists'
    ),
}
INTERPRETATION = (
    'Heuristic evidence score in [0, 1); not a probability of regime change or failure.'
)
METHOD = 'pre_monitoring_regime_signal_mean_sample_std_ddof1'
NOT_EVALUATED = 'not_evaluated'
LEVELS = ('normal', 'watch', 'alert')
DEPRECATED_KEYS = {
    'medium_risk_threshold': 'watch_threshold',
    'high_risk_threshold': 'alert_threshold',
}


@dataclass(frozen=True)
class Level3Parameters:
    """Score thresholds (inclusive) and the consecutive-acquisition requirement."""

    watch_threshold: float = 0.4
    alert_threshold: float = 0.7
    consecutive_acquisitions: int = 2

    def __post_init__(self) -> None:
        for name in ('watch_threshold', 'alert_threshold'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f'level3.{name} must be in (0, 1).')
        if self.watch_threshold >= self.alert_threshold:
            raise ValueError('level3.watch_threshold must be below alert_threshold.')
        if int(self.consecutive_acquisitions) != self.consecutive_acquisitions or (
            self.consecutive_acquisitions < 1
        ):
            raise ValueError('level3.consecutive_acquisitions must be a positive integer.')

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Level3Parameters:
        """Read ``level3``; fall back to deprecated dashboard regime thresholds."""
        settings = config.get('level3', {})
        if not isinstance(settings, dict):
            raise ValueError('level3 must be a mapping.')
        settings = dict(settings)
        regime = config.get('monitoring', {}).get('dashboard', {}).get('regime', {})
        for old, new in DEPRECATED_KEYS.items():
            if isinstance(regime, dict) and old in regime:
                LOGGER.warning(
                    'monitoring.dashboard.regime.%s is deprecated; use level3.%s.',
                    old,
                    new,
                )
                settings.setdefault(new, regime[old])
        defaults = cls()
        return cls(
            watch_threshold=float(
                settings.get('watch_threshold', defaults.watch_threshold)
            ),
            alert_threshold=float(
                settings.get('alert_threshold', defaults.alert_threshold)
            ),
            consecutive_acquisitions=int(
                settings.get('consecutive_acquisitions', defaults.consecutive_acquisitions)
            ),
        )


def z_equivalent(threshold: float) -> float:
    """Unsmoothed, fully persistent regime z that maps to a score threshold."""
    return float(-np.log1p(-threshold))


@dataclass(frozen=True)
class Level3Result:
    """Per-acquisition L3 evidence; levels are ``not_evaluated`` off monitoring."""

    score: np.ndarray
    regional_score: np.ndarray
    evidence_score: np.ndarray
    level: np.ndarray
    criteria: dict[str, np.ndarray]
    first_watch_index: int | None
    first_alert_index: int | None
    parameters: Level3Parameters
    calibration: dict[str, Any]
    calibration_replay: dict[str, Any] | None


def sustained(values: np.ndarray, threshold: float, count: int) -> np.ndarray:
    """True where ``values >= threshold`` held for ``count`` consecutive dates.

    Missing values break a run.
    """
    met = np.isfinite(values) & (values >= threshold)
    output = np.zeros(met.size, dtype=bool)
    run = 0
    for index, value in enumerate(met):
        run = run + 1 if value else 0
        output[index] = run >= count
    return output


def _episodes(flags: np.ndarray) -> int:
    return int(np.sum(flags[1:] & ~flags[:-1]) + (flags[0] if flags.size else 0))


def _first(flags: np.ndarray) -> int | None:
    indices = np.flatnonzero(flags)
    return int(indices[0]) if indices.size else None


def regime_calibration_replay(
    scores: np.ndarray, parameters: Level3Parameters
) -> dict[str, Any] | None:
    """Score-only replay over in-sample pre-monitoring dates.

    The baseline is fitted on these dates, so the counts are optimistic. L2
    and coherence are not available here; only the score criteria are counted.
    """
    finite = np.isfinite(scores)
    if not np.any(finite):
        return None
    count = parameters.consecutive_acquisitions
    return {
        'samples': int(np.sum(finite)),
        'fraction_at_or_above_watch': float(
            np.mean(scores[finite] >= parameters.watch_threshold)
        ),
        'fraction_at_or_above_alert': float(
            np.mean(scores[finite] >= parameters.alert_threshold)
        ),
        'sustained_watch_score_episodes': _episodes(
            sustained(scores, parameters.watch_threshold, count)
        ),
        'sustained_alert_score_episodes': _episodes(
            sustained(scores, parameters.alert_threshold, count)
        ),
        'in_sample': True,
    }


def frozen_level3_calibration(
    path: Path,
    baseline: dict[str, Any],
    dates: tuple[str, ...],
    *,
    unit: str,
    target_feature: str,
) -> dict[str, Any]:
    """Record the pre-monitoring regime baseline once; reject later changes.

    ``baseline`` comes from ``TemporalMonitoringResult.regime_calibration``.
    The fingerprint covers the baseline signal values and their dates.
    """
    values = np.asarray(baseline['baseline_values'], dtype='<f8')
    first, last = int(baseline['first_index']), int(baseline['last_index'])
    digest = hashlib.sha256()
    digest.update(json.dumps(dates[first : last + 1]).encode())
    digest.update(str(baseline['signal']).encode())
    digest.update(values.tobytes())
    fingerprint = digest.hexdigest()
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if (
            not compatible_schema(payload.get('schema_version'))
            or payload.get('method') != METHOD
            or payload.get('signal') != baseline['signal']
            or payload.get('unit') != unit
            or payload.get('target_feature') != target_feature
        ):
            raise ValueError(
                'Frozen L3 calibration has an incompatible method, signal, unit or target.'
            )
        if payload.get('input_fingerprint') != fingerprint:
            raise ValueError(
                'L3 calibration history changed. Use a new calibration_file or run '
                'directory for an explicitly recalibrated model.'
            )
        return payload
    payload = {
        'schema_version': SCHEMA_VERSION,
        'method': METHOD,
        'formula': SCORE_FORMULA,
        'signal': str(baseline['signal']),
        'population': 'tsf_mean_on_fixed_support',
        'unit': unit,
        'signal_unit': f'{unit}/day' if unit else '',
        'target_feature': target_feature,
        'mean': float(baseline['mean']),
        'sigma': positive_finite(baseline['sigma'], 'L3 baseline sigma'),
        'sample_count': int(baseline['sample_count']),
        'first_date': dates[first],
        'last_date': dates[last],
        'input_fingerprint': fingerprint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing another run's calibration.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return payload


def compute_level3(
    score: np.ndarray,
    regional_score: np.ndarray,
    level2: Level2Result,
    monitoring_window: tuple[int, int],
    parameters: Level3Parameters,
    regions: tuple[SpatialCoherenceRegion, ...] = (),
    *,
    calibration: dict[str, Any] | None = None,
    calibration_scores: np.ndarray | None = None,
) -> Level3Result:
    """Apply the L3 evidence rule causally to every monitoring acquisition."""
    score = np.asarray(score, dtype=np.float64)
    time_count = score.size
    start, end = monitoring_window
    if not 0 <= start < end <= time_count:
        raise ValueError('L3 monitoring window is outside the acquisition range.')
    monitoring = np.zeros(time_count, dtype=bool)
    monitoring[start:end] = True
    evidence = np.fmax(score, np.asarray(regional_score, dtype=np.float64))
    evidence = np.where(monitoring, evidence, np.nan)
    count = parameters.consecutive_acquisitions
    watch_score = sustained(evidence, parameters.watch_threshold, count)
    alert_score = sustained(evidence, parameters.alert_threshold, count)
    l2_alarm = monitoring & (level2.alarm | level2.regional_alarm)
    coherent = np.zeros(time_count, dtype=bool)
    for region in regions:
        coherent[max(int(region.activation_index), 0) :] = True
    coherent &= monitoring
    alert = l2_alarm & alert_score & coherent
    watch = monitoring & ~alert & (watch_score | l2_alarm)
    level = np.full(time_count, NOT_EVALUATED, dtype='<U13')
    level[monitoring] = 'normal'
    level[watch] = 'watch'
    level[alert] = 'alert'
    return Level3Result(
        score=score,
        regional_score=np.asarray(regional_score, dtype=np.float64),
        evidence_score=evidence,
        level=level,
        criteria={
            'sustained_watch_score': watch_score,
            'sustained_alert_score': alert_score,
            'l2_alarm': l2_alarm,
            'coherent_region': coherent,
        },
        first_watch_index=_first(watch | alert),
        first_alert_index=_first(alert),
        parameters=parameters,
        calibration=calibration or {},
        calibration_replay=(
            None
            if calibration_scores is None
            else regime_calibration_replay(
                np.asarray(calibration_scores, dtype=np.float64), parameters
            )
        ),
    )


def level3_metadata(result: Level3Result, dates: tuple[str, ...]) -> dict[str, Any]:
    """JSON-ready L3 contract block for monitoring metadata and exports."""
    parameters = result.parameters

    def date_of(index: int | None) -> str | None:
        return None if index is None else dates[index]

    monitored = result.level != NOT_EVALUATED
    return {
        'score_name': 'regime_change_score',
        'score_formula': SCORE_FORMULA,
        'interpretation': INTERPRETATION,
        'evidence_rule': EVIDENCE_RULE,
        'evidence_score': 'max(TSF-wide score, coherent-region score)',
        'watch_threshold': parameters.watch_threshold,
        'alert_threshold': parameters.alert_threshold,
        'threshold_comparison': 'inclusive',
        'consecutive_acquisitions': parameters.consecutive_acquisitions,
        'watch_threshold_z_equivalent': z_equivalent(parameters.watch_threshold),
        'alert_threshold_z_equivalent': z_equivalent(parameters.alert_threshold),
        'calibration': result.calibration,
        'calibration_replay': result.calibration_replay,
        'first_watch_date': date_of(result.first_watch_index),
        'first_alert_date': date_of(result.first_alert_index),
        'level_counts': {
            name: int(np.sum(result.level[monitored] == name)) for name in LEVELS
        },
        'series': [
            {
                'date': dates[index],
                'regime_change_score': _number(result.score[index]),
                'coherent_region_regime_change_score': _number(
                    result.regional_score[index]
                ),
                'l3_level': str(result.level[index]),
                **{
                    f'l3_{name}': bool(flags[index])
                    for name, flags in result.criteria.items()
                },
            }
            for index in np.flatnonzero(monitored)
        ],
    }


def _number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None
