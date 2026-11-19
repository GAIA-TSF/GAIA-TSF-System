"""Common engine interface and the shared alert output schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ale_3levels.contract import ForecastContract

NO_DATA = 'NO_DATA'
NO_DATA_SEVERITY = -1

# Ordinal severity, for comparison between engines only (README: severity mapping).
SEVERITY = {
    'fixed_gates': {'WHITE': 0, 'GREEN': 1, 'YELLOW': 2, 'RED': 3},
    'three_level': {'NONE': 0, 'L1': 1, 'L2': 2, 'L3': 3},
}


@dataclass(frozen=True)
class AlertResult:
    """Per-pixel, per-acquisition output written by every engine.

    ``level`` and ``trigger_reason`` are integer codes into ``levels`` and
    ``trigger_reasons``. ``severity`` is the ordinal 0-3, or -1 for NO_DATA.
    ``recommended_actions`` maps a level label to advisory text from config.
    """

    engine: str
    engine_version: str
    config_hash: str
    dates: np.ndarray
    levels: tuple[str, ...]
    level: np.ndarray
    severity: np.ndarray
    metric_values: dict[str, np.ndarray]
    trigger_reasons: tuple[str, ...]
    trigger_reason: np.ndarray
    recommended_actions: dict[str, list[str]] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine not in SEVERITY:
            raise ValueError(f'Unknown engine {self.engine!r}.')
        if NO_DATA not in self.levels:
            raise ValueError('levels must include NO_DATA.')
        unknown = set(self.levels) - set(SEVERITY[self.engine]) - {NO_DATA}
        if unknown:
            raise ValueError(
                f'Levels {sorted(unknown)} are not native to {self.engine}.'
            )
        shape = self.level.shape
        if len(shape) != 3 or shape[0] != len(self.dates):
            raise ValueError('level must be shaped (time, row, col) matching dates.')
        for name, values in (
            ('severity', self.severity),
            ('trigger_reason', self.trigger_reason),
            *self.metric_values.items(),
        ):
            if values.shape != shape:
                raise ValueError(
                    f'{name} shape {values.shape} differs from level {shape}.'
                )
        if self.level.size and (
            self.level.min() < 0 or self.level.max() >= len(self.levels)
        ):
            raise ValueError('level codes must index levels.')
        expected = severity_from_levels(self.engine, self.levels, self.level)
        if not np.array_equal(expected, self.severity):
            raise ValueError('severity must follow the documented level mapping.')

    def labels(self) -> np.ndarray:
        """Native level labels as a string array."""
        return np.asarray(self.levels)[self.level]


def severity_from_levels(
    engine: str, levels: tuple[str, ...], codes: np.ndarray
) -> np.ndarray:
    """Map native level codes to the documented ordinal severity."""
    mapping = SEVERITY[engine]
    lookup = np.array(
        [NO_DATA_SEVERITY if name == NO_DATA else mapping[name] for name in levels],
        dtype=np.int8,
    )
    return lookup[codes]


@runtime_checkable
class Engine(Protocol):
    """An alert engine: same input contract, same output schema."""

    name: str
    version: str

    def fit(self, calibration_data: ForecastContract) -> None:
        """Freeze calibration parameters (no-op for fixed_gates)."""

    def score(
        self, monitoring_data: ForecastContract, state: Any | None
    ) -> AlertResult:
        """Score monitoring acquisitions; ``state`` holds persisted accumulators."""
