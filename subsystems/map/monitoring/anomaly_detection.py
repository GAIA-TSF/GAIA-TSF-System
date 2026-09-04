"""Configurable statistical anomaly detection from MAP residuals."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.dataset.dataset_builder import Dataset
from subsystems.map.monitoring.residual_analysis import ResidualAnalyzer


@dataclass(frozen=True)
class AnomalyResult:
    """Statistical score and persistent binary anomaly maps."""

    score_stack: np.ndarray
    binary_stack: np.ndarray
    summary: dict[str, Any]


class StatisticalAnomalyDetector:
    """Detect magnitude/z-score residual outliers with temporal persistence."""

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

    def detect(
        self,
        dataset: Dataset,
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
            criteria.append(np.abs(residual_stack) >= self.residual_threshold)
            score_parts.append(np.abs(residual_stack) / self.residual_threshold)
        if self.zscore_threshold is not None:
            criteria.append(np.abs(zscores) >= self.zscore_threshold)
            score_parts.append(np.abs(zscores) / self.zscore_threshold)
        if not criteria:
            raise ValueError('Configure residual_threshold and/or zscore_threshold.')
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
                score_path, result.score_stack[index], dataset, 'anomaly_score'
            )
            ResidualAnalyzer._write_raster(
                binary_path,
                result.binary_stack[index].astype(float),
                dataset,
                'anomaly_binary',
            )
            paths.extend((score_path, binary_path))
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
