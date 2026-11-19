"""Shared calibrated L1 residual anomalies and separate persistence products."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.dataset.dataset_builder import Dataset
from subsystems.map.monitoring.level1 import FORMULA, level1_scores, positive_finite
from subsystems.map.monitoring.residual_analysis import ResidualAnalyzer

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
        dataset: Dataset,
        residual_stack: np.ndarray,
        persistence_start_time_index: int = 0,
        persistence_end_time_index: int | None = None,
        *,
        calibration: dict[str, Any] | None = None,
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

    def write(
        self,
        result: AnomalyResult,
        dataset: Dataset,
        output_dir: Path,
        *,
        residual_rate_unit: str = '',
    ) -> list[Path]:
        """Write score/binary rasters and the JSON anomaly summary."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, date in enumerate(dataset.dates):
            suffix = ResidualAnalyzer._safe_date(date)
            score_path = output_dir / f'anomaly_score_{suffix}.tif'
            binary_path = output_dir / f'anomaly_binary_{suffix}.tif'
            ResidualAnalyzer._write_raster(
                score_path,
                result.score_stack[index],
                dataset,
                'anomaly_score',
                unit='dimensionless',
                time_index=index,
            )
            ResidualAnalyzer._write_raster(
                binary_path,
                np.where(
                    np.isfinite(result.score_stack[index]),
                    result.binary_stack[index].astype(float),
                    np.nan,
                ),
                dataset,
                'persistent_l1_anomaly_binary',
                time_index=index,
            )
            paths.extend((score_path, binary_path))
            if (
                result.zscore_stack is not None
                and result.instantaneous_stack is not None
            ):
                z_path = output_dir / f'residual_z_{suffix}.tif'
                l1_path = output_dir / f'anomaly_l1_binary_{suffix}.tif'
                ResidualAnalyzer._write_raster(
                    z_path,
                    result.zscore_stack[index],
                    dataset,
                    'standardized_residual',
                    unit='dimensionless',
                    time_index=index,
                )
                ResidualAnalyzer._write_raster(
                    l1_path,
                    np.where(
                        np.isfinite(result.zscore_stack[index]),
                        result.instantaneous_stack[index].astype(float),
                        np.nan,
                    ),
                    dataset,
                    'instantaneous_l1_anomaly_binary',
                    time_index=index,
                )
                paths.extend((z_path, l1_path))
        summary = {
            **result.summary,
            'residual_rate_unit': residual_rate_unit,
            'anomaly_score_unit': 'dimensionless',
            'anomaly_binary_unit': 'flag (0 or 1)',
        }
        (output_dir / 'anomaly_summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding='utf-8',
        )
        return paths

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
