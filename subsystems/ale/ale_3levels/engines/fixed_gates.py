"""Engine A, ``fixed_gates``: fixed risk gates on the 30-day LOS movement.

Metric, per pixel and acquisition date ``t`` (window ``W`` days, default 30):

* ``gate_input: observed``: ``disp = d(t) - d(t - W)`` on the observed
  cumulative LOS, with ``d(t - W)`` linearly interpolated between the two
  acquisitions bracketing ``t - W``;
* ``gate_input: residual``: the same window over MAP's one-step residuals,
  ``sum(residual_mm)`` over the forecast intervals inside ``(t - W, t]``, the
  interval containing ``t - W`` pro-rated. This equals the movement of the
  cumulative residual with the same linear interpolation.

``rate_mm_per_month = disp * 30 / W``. ``NO_DATA`` when the acquisition before
``t - W`` is more than ``interpolation_tolerance_days`` away (or absent), when
a bracketing value of the pixel is missing, or outside the TSF mask.

Gates on the directional magnitude ``m`` of ``rate_mm_per_month``
(``abs``: ``|r|``; ``negative``: ``max(0, -r)``, motion away from the
satellite; ``positive``: ``max(0, r)``):

    WHITE  m <  white_below
    GREEN  white_below <= m <  green_below
    YELLOW green_below <= m <= yellow_max
    RED    m >  yellow_max, or velocity doubling

Velocity doubling: RED when ``m(t) >= ratio * m(t - W)`` with
``m(t) >= min_rate_mm`` and ``m(t - W) >= min_prev_rate_mm``; ``m(t - W)`` is
the same metric for the window ending at ``t - W``. When that window is not
available, doubling is skipped and the gate level stands.

The engine is a pure function of the contract and its configuration: it
executes no action and writes nothing. Recommended actions are text only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ale_3levels import __version__
from ale_3levels.config import config_hash
from ale_3levels.contract import ForecastContract
from ale_3levels.engines.base import NO_DATA, AlertResult, severity_from_levels

ENGINE_VERSION = f'{__version__}+fixed_gates.1'
LEVELS = (NO_DATA, 'WHITE', 'GREEN', 'YELLOW', 'RED')
NO_DATA_CODE, WHITE, GREEN, YELLOW, RED = range(5)
GATE_INPUTS = ('observed', 'residual')
DIRECTIONS = ('abs', 'negative', 'positive')

# Reasons: a NO_DATA cause, or "<gate>|<doubling status>".
NO_DATA_REASONS = (
    'no_data:outside_mask',
    'no_data:no_acquisition_within_tolerance',
    'no_data:missing_value',
)
GATE_REASONS = ('white', 'green', 'yellow', 'red_threshold')
DOUBLING_STATUSES = (
    'doubling_fired',
    'doubling_not_met',
    'doubling_below_min_rate',
    'doubling_prior_below_floor',
    'doubling_skipped_no_prior_window',
    'doubling_disabled',
)
TRIGGER_REASONS = NO_DATA_REASONS + tuple(
    f'{gate}|{status}' for gate in GATE_REASONS for status in DOUBLING_STATUSES
)


@dataclass(frozen=True)
class FixedGatesConfig:
    """Validated ``fixed_gates.yaml``."""

    window_days: float
    tolerance_days: float
    gate_input: str
    direction: str
    white_below: float
    green_below: float
    yellow_max: float
    doubling_enabled: bool
    doubling_ratio: float
    doubling_min_rate: float
    doubling_min_prev_rate: float
    recommended_actions: dict[str, list[str]]

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> FixedGatesConfig:
        gates = config['gates']
        doubling = config.get('velocity_doubling') or {}
        value = cls(
            window_days=float(config['window_days']),
            tolerance_days=float(config['interpolation_tolerance_days']),
            gate_input=str(config.get('gate_input', 'observed')),
            direction=str(config.get('direction', 'abs')),
            white_below=float(gates['white_below']),
            green_below=float(gates['green_below']),
            yellow_max=float(gates['yellow_max']),
            doubling_enabled=bool(doubling.get('enabled', True)),
            doubling_ratio=float(doubling.get('ratio', 2.0)),
            doubling_min_rate=float(doubling.get('min_rate_mm', 4.0)),
            doubling_min_prev_rate=float(doubling.get('min_prev_rate_mm', 1.0)),
            recommended_actions={
                str(level): [str(text) for text in texts or []]
                for level, texts in (config.get('recommended_actions') or {}).items()
            },
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.window_days <= 0 or self.tolerance_days < 0:
            raise ValueError('window_days must be > 0 and tolerance >= 0.')
        if self.gate_input not in GATE_INPUTS:
            raise ValueError(f'gate_input must be one of {GATE_INPUTS}.')
        if self.direction not in DIRECTIONS:
            raise ValueError(f'direction must be one of {DIRECTIONS}.')
        if not 0 < self.white_below < self.green_below <= self.yellow_max:
            raise ValueError(
                'Gates must satisfy 0 < white_below < green_below <= yellow_max.'
            )
        if self.doubling_ratio <= 1 or self.doubling_min_rate < 0:
            raise ValueError('velocity_doubling needs ratio > 1 and min_rate_mm >= 0.')
        if self.doubling_min_prev_rate <= 0:
            raise ValueError('velocity_doubling.min_prev_rate_mm must be > 0.')


def _days(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype('<U10').astype('datetime64[D]').astype(np.int64)


def _interpolate_level(
    levels: np.ndarray, days: np.ndarray, x: int, tolerance: float
) -> np.ndarray | None:
    """Linear interpolation of a ``(time, row, col)`` level series at day ``x``.

    Returns None when no acquisition lies within ``tolerance`` before ``x``
    (or none after it). NaN pixels propagate.
    """
    k = int(np.searchsorted(days, x, side='left'))
    if k < days.size and days[k] == x:
        return levels[k]
    if k == 0 or k == days.size or x - days[k - 1] > tolerance:
        return None
    fraction = (x - days[k - 1]) / (days[k] - days[k - 1])
    return levels[k - 1] + fraction * (levels[k] - levels[k - 1])


def _window_sum(
    increments: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    x0: int,
    x1: int,
    tolerance: float,
) -> np.ndarray | None:
    """Sum of interval increments over ``(x0, x1]``, pro-rated at both ends.

    Intervals ``(starts[i], ends[i]]`` must cover ``(x0, x1]`` without gaps,
    and an interval cut by ``x0`` or ``x1`` must start within ``tolerance``
    before the cut (the same rule as interpolation).
    """
    overlap = np.minimum(ends, x1) - np.maximum(starts, x0)
    used = overlap > 0
    if not np.any(used) or int(overlap[used].sum()) != x1 - x0:
        return None
    for cut in (x0, x1):
        cutting = used & (starts < cut) & (ends > cut)
        if np.any(cut - starts[cutting] > tolerance):
            return None
    weights = overlap[used] / (ends[used] - starts[used])
    return np.tensordot(weights, increments[used], axes=(0, 0))


class FixedGatesEngine:
    """Engine A: fixed gates on the 30-day LOS movement."""

    name = 'fixed_gates'
    version = ENGINE_VERSION

    def __init__(self, config: dict[str, Any]) -> None:
        self.raw_config = dict(config)
        self.config = FixedGatesConfig.from_mapping(config)
        self.config_hash = config_hash(self.raw_config)

    def fit(self, calibration_data: ForecastContract) -> None:
        """Fixed thresholds need no calibration."""

    def score(
        self, monitoring_data: ForecastContract, state: Any | None = None
    ) -> AlertResult:
        """Gate every acquisition of the contract (pure; writes nothing)."""
        contract = monitoring_data
        config = self.config
        window = round(config.window_days)
        if window != config.window_days:
            raise ValueError('window_days must be a whole number of days.')
        tolerance = config.tolerance_days
        scale = 30.0 / config.window_days
        dates = contract.dates
        target_days = _days(dates)
        if config.gate_input == 'observed':
            levels = contract.observed_series_mm
            level_days = _days(contract.series_dates)

            def movement(end: int) -> np.ndarray | None:
                at_end = _interpolate_level(levels, level_days, end, tolerance)
                at_start = _interpolate_level(
                    levels, level_days, end - window, tolerance
                )
                if at_end is None or at_start is None:
                    return None
                return at_end - at_start

        else:
            if contract.lead != 1:
                raise ValueError('gate_input: residual needs lead-1 residuals.')
            # Lead-1 residual i covers (forecast origin i, target date i].
            ends = target_days
            starts = _days(contract.forecast_origin)
            increments = contract.residual_mm

            def movement(end: int) -> np.ndarray | None:
                return _window_sum(
                    increments, starts, ends, end - window, end, tolerance
                )

        shape = contract.observed_mm.shape
        disp = np.full(shape, np.nan)
        prev_disp = np.full(shape, np.nan)
        prior_available = np.zeros(shape[0], dtype=bool)
        window_available = np.zeros(shape[0], dtype=bool)
        for index, end in enumerate(target_days):
            current = movement(int(end))
            if current is None:
                continue
            window_available[index] = True
            disp[index] = current
            previous = movement(int(end) - window)
            if previous is not None:
                prior_available[index] = True
                prev_disp[index] = previous

        rate = disp * scale
        prev_rate = prev_disp * scale
        magnitude = self._directional(rate)
        prev_magnitude = self._directional(prev_rate)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(prev_magnitude > 0, magnitude / prev_magnitude, np.nan)

        inside = np.broadcast_to(contract.mask, shape)
        finite = np.isfinite(magnitude) & inside
        level = np.full(shape, NO_DATA_CODE, dtype=np.int8)
        gate = np.full(shape, -1, dtype=np.int8)
        gate[finite & (magnitude < config.white_below)] = 0
        gate[
            finite
            & (magnitude >= config.white_below)
            & (magnitude < config.green_below)
        ] = 1
        gate[
            finite
            & (magnitude >= config.green_below)
            & (magnitude <= config.yellow_max)
        ] = 2
        gate[finite & (magnitude > config.yellow_max)] = 3
        level[gate >= 0] = gate[gate >= 0] + WHITE

        doubling = self._doubling_status(
            finite, magnitude, prev_magnitude, prior_available[:, None, None]
        )
        fired = doubling == DOUBLING_STATUSES.index('doubling_fired')
        level[fired] = RED

        reason = np.zeros(shape, dtype=np.int16)
        gate_offset = len(NO_DATA_REASONS)
        reason[finite] = (
            gate_offset
            + gate[finite].astype(np.int16) * len(DOUBLING_STATUSES)
            + doubling[finite]
        )
        no_window = ~window_available[:, None, None] & inside
        reason[~inside] = NO_DATA_REASONS.index('no_data:outside_mask')
        reason[no_window] = NO_DATA_REASONS.index(
            'no_data:no_acquisition_within_tolerance'
        )
        reason[inside & window_available[:, None, None] & ~finite] = (
            NO_DATA_REASONS.index('no_data:missing_value')
        )

        nan_out = ~finite
        metric_values = {
            'disp_30d_mm': np.where(nan_out, np.nan, disp),
            'rate_mm_per_month': np.where(nan_out, np.nan, rate),
            'velocity_ratio': np.where(nan_out, np.nan, ratio),
            'prev_rate_mm_per_month': np.where(nan_out, np.nan, prev_rate),
        }
        return AlertResult(
            engine=self.name,
            engine_version=self.version,
            config_hash=self.config_hash,
            dates=dates,
            levels=LEVELS,
            level=level,
            severity=severity_from_levels(self.name, LEVELS, level),
            metric_values=metric_values,
            trigger_reasons=TRIGGER_REASONS,
            trigger_reason=reason,
            recommended_actions=config.recommended_actions,
            attrs={
                'gate_input': config.gate_input,
                'direction': config.direction,
                'window_days': config.window_days,
                'interpolation_tolerance_days': config.tolerance_days,
                'gates': {
                    'WHITE': f'm < {config.white_below}',
                    'GREEN': f'{config.white_below} <= m < {config.green_below}',
                    'YELLOW': f'{config.green_below} <= m <= {config.yellow_max}',
                    'RED': f'm > {config.yellow_max} or velocity doubling',
                },
                'gated_quantity': 'directional magnitude of rate_mm_per_month',
                'velocity_doubling': {
                    'enabled': config.doubling_enabled,
                    'ratio': config.doubling_ratio,
                    'min_rate_mm': config.doubling_min_rate,
                    'min_prev_rate_mm': config.doubling_min_prev_rate,
                },
                'level_scope': 'pixel',
            },
        )

    def _directional(self, rate: np.ndarray) -> np.ndarray:
        if self.config.direction == 'abs':
            return np.abs(rate)
        if self.config.direction == 'negative':
            return np.maximum(0.0, -rate)
        return np.maximum(0.0, rate)

    def _doubling_status(
        self,
        finite: np.ndarray,
        magnitude: np.ndarray,
        prev_magnitude: np.ndarray,
        prior_available: np.ndarray,
    ) -> np.ndarray:
        config = self.config
        index = DOUBLING_STATUSES.index
        status = np.full(magnitude.shape, index('doubling_disabled'), dtype=np.int16)
        if not config.doubling_enabled:
            return status
        with np.errstate(invalid='ignore'):
            has_prior = prior_available & np.isfinite(prev_magnitude)
            status[finite & ~has_prior] = index('doubling_skipped_no_prior_window')
            candidate = finite & has_prior
            below_min = candidate & (magnitude < config.doubling_min_rate)
            prior_floor = (
                candidate
                & ~below_min
                & (prev_magnitude < config.doubling_min_prev_rate)
            )
            tested = candidate & ~below_min & ~prior_floor
            fired = tested & (magnitude >= config.doubling_ratio * prev_magnitude)
        status[below_min] = index('doubling_below_min_rate')
        status[prior_floor] = index('doubling_prior_below_floor')
        status[tested] = index('doubling_not_met')
        status[fired] = index('doubling_fired')
        return status
