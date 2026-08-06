"""Residual calculation, statistics and spatial product writing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

try:
    import rasterio
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installs
    rasterio = None

from subsystems.map.dataset.dataset_builder import Dataset


@dataclass(frozen=True)
class ResidualResult:
    """Residual samples, restored time stack and descriptive statistics."""

    values: np.ndarray
    stack: np.ndarray
    statistics: dict[str, float | int]


class ResidualAnalyzer:
    """Compute and persist observation-minus-prediction products."""

    def analyze(self, dataset: Dataset, predictions: np.ndarray) -> ResidualResult:
        """Calculate residuals and map the values back into the raster grid."""
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.shape != dataset.targets.shape:
            raise ValueError('Prediction and observation shapes do not match.')
        residuals = dataset.targets - predictions
        stack = self.restore_stack(dataset, residuals)
        finite = residuals[np.isfinite(residuals)]
        if finite.size == 0:
            raise ValueError('Residual analysis received no finite values.')
        statistics: dict[str, float | int] = {
            'count': int(finite.size),
            'mean': float(np.mean(finite)),
            'std': float(np.std(finite)),
            'min': float(np.min(finite)),
            'max': float(np.max(finite)),
            'mae': float(np.mean(np.abs(finite))),
            'rmse': float(np.sqrt(np.mean(np.square(finite)))),
        }
        return ResidualResult(residuals, stack, statistics)

    def write(
        self, result: ResidualResult, dataset: Dataset, output_dir: Path
    ) -> list[Path]:
        """Write one residual GeoTIFF per acquisition and aggregate statistics JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, date in enumerate(dataset.dates):
            path = output_dir / f'residual_{self._safe_date(date)}.tif'
            self._write_raster(path, result.stack[index], dataset, 'residual')
            paths.append(path)
        (output_dir / 'residual_statistics.json').write_text(
            json.dumps(result.statistics, indent=2, sort_keys=True),
            encoding='utf-8',
        )
        return paths

    @staticmethod
    def restore_stack(dataset: Dataset, values: np.ndarray) -> np.ndarray:
        """Restore flat sample values to a ``(time, rows, columns)`` float stack."""
        stack = np.full(
            (len(dataset.dates), dataset.grid.height, dataset.grid.width), np.nan
        )
        rows = dataset.pixel_indices // dataset.grid.width
        columns = dataset.pixel_indices % dataset.grid.width
        stack[dataset.time_indices, rows, columns] = values
        return stack

    @staticmethod
    def _safe_date(value: str) -> str:
        return value.replace('-', '').replace(':', '').replace(' ', '_')

    @staticmethod
    def _write_raster(
        path: Path, values: np.ndarray, dataset: Dataset, description: str
    ) -> None:
        if rasterio is None:
            raise RuntimeError(
                'Residual product writing requires the Rasterio dependency.'
            )
        nodata = dataset.grid.nodata if dataset.grid.nodata is not None else np.nan
        output = np.where(np.isfinite(values), values, nodata).astype('float32')
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=dataset.grid.height,
            width=dataset.grid.width,
            count=1,
            dtype='float32',
            crs=dataset.grid.crs,
            transform=dataset.grid.transform,
            nodata=nodata,
        ) as raster:
            raster.write(output, 1)
            raster.set_band_description(1, description)
