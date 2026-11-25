"""L1 residual anomalies: frozen-sigma (prototype) and legacy global z-score.

Moved from MAP ``monitoring/anomaly_detection.py`` (f11c5f2 + stash@{0}). The
GeoTIFF writers were removed (ALE writes its own outputs); the computations are
unchanged. ``LegacyGlobalZScoreDetector`` is the 950702d version (Z-A).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from ale_3levels.engines.three_level.grid import GridDataset
from ale_3levels.engines.three_level.level1_zscore import (
    FORMULA,
    level1_scores,
    positive_finite,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyResult:
    """Statistical score and persistent binary anomaly maps."""

    score_stack: np.ndarray
    binary_stack: np.ndarray
    summary: dict[str, Any]
    zscore_stack: np.ndarray | None = None
    instantaneous_stack: np.ndarray | None = None


class StatisticalAnomalyDetector:
    """Detect instantaneous L1 and persistence-qualified standardized residuals."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.residual_threshold = self._optional_positive(
            config.get('residual_threshold')
        )
        self.zscore_threshold = positive_finite(
            config.get('zscore_threshold', 3.0), 'L1 zscore_threshold'
        )
        self.persistence = int(config.get('persistence', 1))
        if self.persistence < 1:
            raise ValueError('anomaly_detection.persistence must be at least one.')
        self.direction = str(config.get('direction', 'both')).lower()
        if self.direction not in {'both', 'positive', 'negative'}:
            raise ValueError(
                "anomaly_detection.direction must be 'both', 'positive', or 'negative'."
            )
        if 'combination_operator' in config:
            raise ValueError(
                'L1 uses only standardized residuals; remove legacy combination_operator.'
            )
        if self.residual_threshold is not None:
            LOGGER.warning(
                'residual_threshold is retained in metadata only; '
                'canonical L1 uses zscore_threshold.'
            )

    def detect(
        self,
        dataset: GridDataset,
        residual_stack: np.ndarray,
        persistence_start_time_index: int = 0,
        persistence_end_time_index: int | None = None,
        *,
        calibration: dict[str, Any] | None = None,
        center: float = 0.0,
    ) -> AnomalyResult:
        """Return scores and persistent binary rasters for monitored pixels.

        Args:
            dataset: MAP dataset that supplies dates and the TSF mask.
            residual_stack: Residuals shaped ``(time, rows, columns)``.
            persistence_start_time_index: First acquisition eligible for persistent
                anomaly analysis. Earlier acquisitions reset the persistence run.
            persistence_end_time_index: Exclusive final eligible acquisition index.
        """
        time_count = residual_stack.shape[0]
        if not 0 <= persistence_start_time_index <= time_count:
            raise ValueError(
                'persistence_start_time_index must be within the residual time range.',
            )
        if persistence_end_time_index is None:
            persistence_end_time_index = time_count
        if not persistence_start_time_index < persistence_end_time_index <= time_count:
            raise ValueError(
                'persistence_end_time_index must be after the persistence start.',
            )
        if calibration is None:
            raise ValueError('L1 detection requires a frozen calibration artifact.')
        if (
            not calibration.get('last_date')
            or calibration['last_date'] >= dataset.dates[persistence_start_time_index]
        ):
            raise ValueError('L1 calibration must end before monitoring starts.')
        if residual_stack.shape != (len(dataset.dates), *dataset.mask.shape):
            raise ValueError('Residual stack does not match dataset dates/mask.')
        eligible = np.zeros((time_count, 1, 1), dtype=bool)
        eligible[persistence_start_time_index:persistence_end_time_index] = True
        residual_stack = np.where(dataset.mask[np.newaxis], residual_stack, np.nan)
        zscores, initial = level1_scores(
            residual_stack,
            calibration['sigma'],
            self.zscore_threshold,
            self.direction,
            eligible,
            center=center,
        )
        binary = self._persistent(initial)
        directional_z = (
            -zscores
            if self.direction == 'negative'
            else zscores
            if self.direction == 'positive'
            else np.abs(zscores)
        )
        score = np.maximum(0.0, directional_z) / self.zscore_threshold
        by_time = {
            date: int(np.count_nonzero(binary[index]))
            for index, date in enumerate(dataset.dates)
        }
        summary = {
            'direction': self.direction,
            'formula': FORMULA,
            'comparison': 'strict',
            'calibration': calibration,
            'residual_mean': calibration.get('mean_residual_diagnostic'),
            'residual_std': calibration['sigma'],
            'legacy_residual_threshold_diagnostic': self.residual_threshold,
            'zscore_threshold': self.zscore_threshold,
            'persistence': self.persistence,
            'persistence_start_time_index': persistence_start_time_index,
            'persistence_start_date': (
                dataset.dates[persistence_start_time_index]
                if persistence_start_time_index < len(dataset.dates)
                else None
            ),
            'persistence_end_time_index': persistence_end_time_index,
            'persistence_end_date': dataset.dates[persistence_end_time_index - 1],
            'anomalous_pixels_by_acquisition': by_time,
            'total_anomalous_samples': int(np.count_nonzero(binary)),
            'instantaneous_l1_pixels_by_acquisition': {
                date: int(np.count_nonzero(initial[index]))
                for index, date in enumerate(dataset.dates)
            },
            'anomaly_binary_semantics': 'persistence_qualified_l1',
            'anomaly_score_semantics': 'directional_z_divided_by_threshold',
        }
        return AnomalyResult(score, binary, summary, zscores, initial)

    @staticmethod
    def _optional_positive(value: object) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError('Anomaly thresholds must be finite and positive.')
        return numeric

    def _persistent(self, values: np.ndarray) -> np.ndarray:
        if self.persistence == 1:
            return values
        output = np.zeros_like(values, dtype=bool)
        run = np.zeros(values.shape[1:], dtype=np.int64)
        for index in range(values.shape[0]):
            run = np.where(values[index], run + 1, 0)
            output[index] = run >= self.persistence
        return output


class LegacyGlobalZScoreDetector:
    """Z-A: the 950702d L1 detector, kept verbatim for comparison.

    ``z = (r - mean(R)) / std(R)`` over *every* finite residual in the stack,
    including monitoring dates (so earlier z change when data are appended),
    OR/AND-combined with an absolute ``residual_threshold``. Selected by
    ``zscore_formula: legacy_global``. Only ``write`` (GeoTIFF I/O) is omitted.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.residual_threshold = self._optional_positive(
            config.get('residual_threshold')
        )
        self.zscore_threshold = self._optional_positive(
            config.get('zscore_threshold', 3.0)
        )
        self.persistence = int(config.get('persistence', 1))
        if self.persistence < 1:
            raise ValueError('anomaly_detection.persistence must be at least one.')
        self.direction = str(config.get('direction', 'both')).lower()
        if self.direction not in {'both', 'positive', 'negative'}:
            raise ValueError(
                "anomaly_detection.direction must be 'both', 'positive', or 'negative'."
            )
        self.combination_operator = str(
            config.get('combination_operator', 'or')
        ).lower()
        if self.combination_operator not in {'or', 'and'}:
            raise ValueError(
                "anomaly_detection.combination_operator must be 'or' or 'and'."
            )

    def detect(
        self,
        dataset: GridDataset,
        residual_stack: np.ndarray,
        persistence_start_time_index: int = 0,
        persistence_end_time_index: int | None = None,
    ) -> AnomalyResult:
        """Return scores and persistent binary rasters for monitored pixels.

        Args:
            dataset: MAP dataset that supplies dates and the TSF mask.
            residual_stack: Residuals shaped ``(time, rows, columns)``.
            persistence_start_time_index: First acquisition eligible for persistent
                anomaly analysis. Earlier acquisitions reset the persistence run.
            persistence_end_time_index: Exclusive final eligible acquisition index.
        """
        time_count = residual_stack.shape[0]
        if not 0 <= persistence_start_time_index <= time_count:
            raise ValueError(
                'persistence_start_time_index must be within the residual time range.',
            )
        if persistence_end_time_index is None:
            persistence_end_time_index = time_count
        if not persistence_start_time_index < persistence_end_time_index <= time_count:
            raise ValueError(
                'persistence_end_time_index must be after the persistence start.',
            )
        finite = residual_stack[np.isfinite(residual_stack)]
        if finite.size == 0:
            raise ValueError('Cannot detect anomalies without finite residuals.')
        mean, std = float(np.mean(finite)), float(np.std(finite))
        zscores = np.divide(
            residual_stack - mean,
            std,
            out=np.zeros_like(residual_stack),
            where=std > 0,
        )
        criteria: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        if self.residual_threshold is not None:
            if self.direction == 'negative':
                criteria.append(residual_stack <= -self.residual_threshold)
                score_parts.append(
                    np.maximum(0.0, -residual_stack / self.residual_threshold)
                )
            elif self.direction == 'positive':
                criteria.append(residual_stack >= self.residual_threshold)
                score_parts.append(
                    np.maximum(0.0, residual_stack / self.residual_threshold)
                )
            else:
                criteria.append(np.abs(residual_stack) >= self.residual_threshold)
                score_parts.append(np.abs(residual_stack) / self.residual_threshold)
        if self.zscore_threshold is not None:
            if self.direction == 'negative':
                criteria.append(zscores <= -self.zscore_threshold)
                score_parts.append(np.maximum(0.0, -zscores / self.zscore_threshold))
            elif self.direction == 'positive':
                criteria.append(zscores >= self.zscore_threshold)
                score_parts.append(np.maximum(0.0, zscores / self.zscore_threshold))
            else:
                criteria.append(np.abs(zscores) >= self.zscore_threshold)
                score_parts.append(np.abs(zscores) / self.zscore_threshold)
        if not criteria:
            raise ValueError('Configure residual_threshold and/or zscore_threshold.')
        if self.combination_operator == 'and':
            initial = np.logical_and.reduce(criteria) & dataset.mask[np.newaxis, :, :]
        else:
            initial = np.logical_or.reduce(criteria) & dataset.mask[np.newaxis, :, :]
        initial[:persistence_start_time_index] = False
        initial[persistence_end_time_index:] = False
        binary = self._persistent(initial)
        score = np.where(
            dataset.mask[np.newaxis, :, :], np.maximum.reduce(score_parts), np.nan
        )
        binary = binary & dataset.mask[np.newaxis, :, :]
        by_time = {
            date: int(np.count_nonzero(binary[index]))
            for index, date in enumerate(dataset.dates)
        }
        summary = {
            'direction': self.direction,
            'combination_operator': self.combination_operator,
            'residual_mean': mean,
            'residual_std': std,
            'residual_threshold': self.residual_threshold,
            'zscore_threshold': self.zscore_threshold,
            'persistence': self.persistence,
            'persistence_start_time_index': persistence_start_time_index,
            'persistence_start_date': (
                dataset.dates[persistence_start_time_index]
                if persistence_start_time_index < len(dataset.dates)
                else None
            ),
            'persistence_end_time_index': persistence_end_time_index,
            'persistence_end_date': dataset.dates[persistence_end_time_index - 1],
            'anomalous_pixels_by_acquisition': by_time,
            'total_anomalous_samples': int(np.count_nonzero(binary)),
        }
        return AnomalyResult(score, binary, summary)

    @staticmethod
    def _optional_positive(value: object) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if numeric <= 0:
            raise ValueError('Anomaly thresholds must be positive.')
        return numeric

    def _persistent(self, values: np.ndarray) -> np.ndarray:
        if self.persistence == 1:
            return values
        output = np.zeros_like(values, dtype=bool)
        run = np.zeros(values.shape[1:], dtype=np.int16)
        for index in range(values.shape[0]):
            run = np.where(values[index], run + 1, 0)
            output[index] = run >= self.persistence
        return output
