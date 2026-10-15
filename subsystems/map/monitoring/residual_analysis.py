"""Residual calculation, statistics and spatial product writing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
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
        self,
        result: ResidualResult,
        dataset: Dataset,
        output_dir: Path,
        *,
        native_unit: str = '',
        display_unit: str = '',
        value_scale: float = 1.0,
    ) -> list[Path]:
        """Write native residual rasters and unit-explicit statistics JSON."""
        if value_scale <= 0:
            raise ValueError('Residual statistics value_scale must be positive.')
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, date in enumerate(dataset.dates):
            path = output_dir / f'residual_{self._safe_date(date)}.tif'
            self._write_raster(
                path,
                result.stack[index],
                dataset,
                'residual_rate',
                unit=native_unit,
            )
            paths.append(path)
        display_statistics = {
            key: value if key == 'count' else float(value) * value_scale
            for key, value in result.statistics.items()
        }
        payload = {
            'native_rate_unit': native_unit,
            'display_rate_unit': display_unit,
            'value_scale_to_display_unit': value_scale,
            'statistics_native': result.statistics,
            'statistics_display': display_statistics,
        }
        (output_dir / 'residual_statistics.json').write_text(
            json.dumps(payload, indent=2, sort_keys=True),
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
        path: Path,
        values: np.ndarray,
        dataset: Dataset,
        description: str,
        *,
        unit: str = '',
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
            if unit:
                raster.update_tags(1, units=unit)
        ResidualAnalyzer._write_raster_sidecar(
            path,
            dataset,
            description,
            unit=unit,
            nodata=nodata,
        )

    @staticmethod
    def _write_raster_sidecar(
        path: Path,
        dataset: Dataset,
        description: str,
        *,
        unit: str,
        nodata: float | int | None,
    ) -> Path:
        """Write a STAC-style JSON item next to a MAP GeoTIFF product."""
        if rasterio is None:
            raise RuntimeError(
                'Residual product writing requires the Rasterio dependency.'
            )
        west, south, east, north = array_bounds(
            dataset.grid.height,
            dataset.grid.width,
            dataset.grid.transform,
        )
        crs = rasterio.crs.CRS.from_user_input(dataset.grid.crs)
        if crs.is_geographic:
            bbox = [west, south, east, north]
        else:
            bbox = list(transform_bounds(crs, 'EPSG:4326', west, south, east, north))
        properties: dict[str, Any] = {
            'proj:epsg': crs.to_epsg(),
            'proj:shape': [dataset.grid.height, dataset.grid.width],
            'proj:transform': list(dataset.grid.transform)[:6],
            'raster:bands': [{
                'name': description,
                'data_type': 'float32',
                'nodata': _json_number(nodata),
                **({'unit': unit} if unit else {}),
            }],
            'gaia:product_type': description,
        }
        payload = {
            'type': 'Feature',
            'stac_version': '1.0.0',
            'id': path.stem,
            'bbox': bbox,
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[
                    [bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]],
                    [bbox[0], bbox[1]],
                ]],
            },
            'properties': properties,
            'assets': {
                'data': {
                    'href': f'./{path.name}',
                    'type': 'image/tiff; application=geotiff',
                    'roles': ['data'],
                },
            },
            'collection': 'gaia-tsf-map',
        }
        sidecar = path.with_suffix('.json')
        sidecar.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding='utf-8',
        )
        return sidecar


def _json_number(value: float | int | None) -> float | int | None:
    """Convert nodata metadata to a strict JSON value."""
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else numeric
