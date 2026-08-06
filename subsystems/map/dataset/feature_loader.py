"""Loading of engineered raster features created by the DAG subsystem."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import rasterio
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installs
    rasterio = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RasterGrid:
    """Spatial metadata required to write derived MAP rasters."""

    crs: object
    transform: object
    height: int
    width: int
    nodata: float | int | None


@dataclass(frozen=True)
class LoadedFeatures:
    """Aligned feature stacks and their spatial metadata."""

    features: dict[str, np.ndarray]
    dates: tuple[str, ...]
    grid: RasterGrid
    mask: np.ndarray


class FeatureLoader:
    """Load named DAG GeoTIFF outputs without encoding feature names in code."""

    def __init__(self, feature_directories: Iterable[Path], mask_path: Path) -> None:
        self.feature_directories = tuple(feature_directories)
        self.mask_path = mask_path

    def load(self, feature_names: list[str]) -> LoadedFeatures:
        """Load and spatially align each requested feature and the TSF mask."""
        if rasterio is None:
            raise RuntimeError(
                'FeatureLoader requires the Rasterio dependency for GeoTIFF input.'
            )
        if not feature_names:
            raise ValueError('At least one engineered feature must be configured.')
        arrays: dict[str, np.ndarray] = {}
        grid: RasterGrid | None = None
        dates: tuple[str, ...] | None = None
        temporal_dates: tuple[str, ...] | None = None
        for name in feature_names:
            path = self._find_feature(name)
            with rasterio.open(path) as source:
                values = source.read(out_dtype='float64')
                values = self._replace_nodata(values, source.nodata)
                current_grid = RasterGrid(
                    source.crs,
                    source.transform,
                    source.height,
                    source.width,
                    source.nodata,
                )
                current_dates = self._dates_for(path, source.descriptions)
            if grid is None:
                grid, dates = current_grid, current_dates
            else:
                self._validate_grid(grid, current_grid, path)
            if values.shape[0] > 1:
                if (
                    temporal_dates is not None
                    and len(temporal_dates) != values.shape[0]
                ):
                    raise ValueError(
                        f'Feature {name} has incompatible time bands: {path}'
                    )
                temporal_dates = current_dates
            arrays[name] = values
        assert grid is not None and dates is not None
        stack_count = max(values.shape[0] for values in arrays.values())
        if stack_count == 1:
            raise ValueError(
                'MAP temporal splitting requires temporal feature rasters with multiple bands.',
            )
        arrays = {
            name: self._broadcast(values, stack_count, name)
            for name, values in arrays.items()
        }
        with rasterio.open(self.mask_path) as source:
            mask = source.read(1).astype(bool)
            if mask.shape != (grid.height, grid.width):
                raise ValueError(
                    'TSF mask dimensions do not match engineered features.'
                )
        return LoadedFeatures(
            arrays,
            self._normalise_dates(temporal_dates or dates, stack_count),
            grid,
            mask,
        )

    def _find_feature(self, name: str) -> Path:
        for directory in self.feature_directories:
            candidate = directory / f'{name}.tif'
            if candidate.exists():
                return candidate
        searched = ', '.join(str(path) for path in self.feature_directories)
        raise FileNotFoundError(f"Feature '{name}' was not found in: {searched}")

    def _dates_for(
        self, path: Path, descriptions: tuple[str | None, ...]
    ) -> tuple[str, ...]:
        metadata = path.parent / 'metadata.json'
        if metadata.exists():
            try:
                payload = json.loads(metadata.read_text(encoding='utf-8'))
                values = payload.get('acquisition_dates')
                if isinstance(values, list) and values:
                    return tuple(str(value) for value in values)
            except (OSError, json.JSONDecodeError):
                LOGGER.warning('Could not read temporal metadata from %s', metadata)
        return tuple(
            value or f'acquisition_{index:04d}'
            for index, value in enumerate(descriptions)
        )

    @staticmethod
    def _replace_nodata(values: np.ndarray, nodata: float | int | None) -> np.ndarray:
        if nodata is None or not np.isfinite(nodata):
            return values
        return np.where(values == nodata, np.nan, values)

    @staticmethod
    def _validate_grid(
        reference: RasterGrid, candidate: RasterGrid, path: Path
    ) -> None:
        if (reference.height, reference.width, reference.transform, reference.crs) != (
            candidate.height,
            candidate.width,
            candidate.transform,
            candidate.crs,
        ):
            raise ValueError(f'Feature grid does not match first feature: {path}')

    @staticmethod
    def _broadcast(values: np.ndarray, count: int, name: str) -> np.ndarray:
        if values.shape[0] == count:
            return values
        if values.shape[0] == 1:
            return np.broadcast_to(values, (count, *values.shape[1:])).copy()
        raise ValueError(f'Feature {name} cannot be aligned to {count} time steps.')

    @staticmethod
    def _normalise_dates(dates: tuple[str, ...], count: int) -> tuple[str, ...]:
        if len(dates) == count:
            return dates
        return tuple(f'acquisition_{index:04d}' for index in range(count))
