"""Contract v1.0 quality states per acquisition and quality flags per run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from ale_3levels.engines.three_level.calibration import RETROSPECTIVE

LOGGER = logging.getLogger(__name__)

# Evaluated in this order; the first matching state wins.
STATES = (
    'not_monitored',
    'warm_up',
    'missing_acquisition',
    'insufficient_coverage',
    'ok',
)


@dataclass(frozen=True)
class QualityParameters:
    """Coverage and calibration-length limits from the ``quality`` section."""

    minimum_coverage: float = 0.5
    minimum_calibration_samples: int = 20

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError('quality.minimum_coverage must be in (0, 1].')
        if self.minimum_calibration_samples < 2:
            raise ValueError('quality.minimum_calibration_samples must be at least 2.')

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QualityParameters:
        settings = config.get('quality', {})
        if not isinstance(settings, dict):
            raise ValueError('quality must be a mapping.')  # noqa: TRY004 (prototype)
        defaults = cls()
        return cls(
            minimum_coverage=float(
                settings.get('minimum_coverage', defaults.minimum_coverage)
            ),
            minimum_calibration_samples=int(
                settings.get(
                    'minimum_calibration_samples',
                    defaults.minimum_calibration_samples,
                )
            ),
        )


def acquisition_quality(
    observed_stack: np.ndarray,
    prediction_stack: np.ndarray,
    mask: np.ndarray,
    monitoring_window: tuple[int, int],
    parameters: QualityParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-acquisition quality states and observed TSF coverage."""
    support = np.broadcast_to(mask, observed_stack.shape)
    pixels = max(int(np.count_nonzero(mask)), 1)
    coverage = np.sum(support & np.isfinite(observed_stack), axis=(1, 2)) / pixels
    forecast = np.any(support & np.isfinite(prediction_stack), axis=(1, 2))
    states = np.full(observed_stack.shape[0], 'ok', dtype='<U21')
    states[coverage < parameters.minimum_coverage] = 'insufficient_coverage'
    states[coverage == 0] = 'missing_acquisition'
    states[~forecast] = 'warm_up'
    monitored = np.zeros(states.size, dtype=bool)
    monitored[monitoring_window[0] : monitoring_window[1]] = True
    states[~monitored] = 'not_monitored'
    return states, coverage


def calibration_date_counts(
    l1: dict[str, Any], l2: dict[str, Any], l3: dict[str, Any]
) -> dict[str, int]:
    """Pre-monitoring calibration dates per level (L1 pools pixels per date)."""
    return {
        'l1': len(l1.get('calibration_dates') or []) or int(l2['sample_count']),
        'l2': int(l2['sample_count']),
        'l3': int(l3['sample_count']),
    }


def run_quality_flags(
    *,
    run_mode: str,
    l2_bias_warning: bool,
    calibration_dates: dict[str, int],
    calibrations_reused: list[str],
    parameters: QualityParameters,
) -> list[dict[str, Any]]:
    """Return run-level quality flags and log a warning for each one."""
    flags: list[dict[str, Any]] = []
    if run_mode == RETROSPECTIVE:
        flags.append(
            {
                'flag': 'retrospective_training',
                'detail': 'training used monitoring-period information',
            }
        )
    if l2_bias_warning:
        flags.append(
            {
                'flag': 'l2_calibration_bias',
                'detail': '|mean residual / sigma| exceeds level2.bias_warning_ratio',
            }
        )
    short = {
        level: count
        for level, count in calibration_dates.items()
        if count < parameters.minimum_calibration_samples
    }
    if short:
        flags.append(
            {
                'flag': 'short_calibration',
                'detail': (
                    f'fewer than {parameters.minimum_calibration_samples} '
                    'pre-monitoring calibration dates; sigma and baselines are '
                    'uncertain and early score-only L3 watch is less reliable'
                ),
                'calibration_dates': short,
            }
        )
    if calibrations_reused:
        flags.append(
            {
                'flag': 'calibration_reused',
                'detail': 'frozen calibration artifacts existed before this run',
                'levels': calibrations_reused,
            }
        )
    for flag in flags:
        LOGGER.warning('Quality flag %s: %s.', flag['flag'], flag['detail'])
    return flags
