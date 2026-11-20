"""Loading of engineered raster features created by the DAG subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
    """Load and causally align named engineered DAG feature rasters."""

    def __init__(
        self,
        feature_directories: Iterable[Path],
        mask_path: Path,
        temporal_alignment_method: str = 'exact',
    ) -> None:
        self.feature_directories = tuple(feature_directories)
        self.mask_path = mask_path
        if temporal_alignment_method not in {'exact', 'previous'}:
            raise ValueError(
                'temporal_alignment_method must be "exact" or "previous".',
            )
        self.temporal_alignment_method = temporal_alignment_method

    def load(
        self,
        feature_names: list[str],
        reference_feature: str | None = None,
    ) -> LoadedFeatures:
        """Load feature rasters and align every stack to a temporal reference.

        Meteorological products may be daily while the deformation target is
        acquired less frequently. Their values are therefore selected by their
        acquisition date instead of being aligned by raster-band number.

        Args:
            feature_names: DAG feature stems selected by the MAP configuration.
            reference_feature: Multi-band feature whose dates define the model
                time axis. For MAP learning this is the target feature.
        """
        if rasterio is None:
            raise RuntimeError(
                'FeatureLoader requires the Rasterio dependency for GeoTIFF input.'
            )
        if not feature_names:
            raise ValueError('At least one engineered feature must be configured.')
        arrays: dict[str, np.ndarray] = {}
        dates_by_feature: dict[str, tuple[str, ...]] = {}
        grid: RasterGrid | None = None
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
                grid = current_grid
            else:
                self._validate_grid(grid, current_grid, path)
            arrays[name] = values
            dates_by_feature[name] = current_dates
        assert grid is not None

        reference_name = reference_feature or feature_names[0]
        if reference_name not in arrays:
            raise KeyError(f'Reference feature was not loaded: {reference_name}')
        reference_values = arrays[reference_name]
        reference_dates = dates_by_feature[reference_name]
        if reference_values.shape[0] == 1:
            raise ValueError(
                'MAP temporal splitting requires a multi-band reference feature.',
            )
        if len(reference_dates) != reference_values.shape[0]:
            raise ValueError(
                f'Reference feature {reference_name} has incompatible dates and bands.',
            )
        arrays = {
            name: self._align_to_reference_dates(
                values,
                dates_by_feature[name],
                reference_dates,
                name,
            )
            for name, values in arrays.items()
        }
        with rasterio.open(self.mask_path) as source:
            mask = source.read(1).astype(bool)
            mask_grid = RasterGrid(
                source.crs,
                source.transform,
                source.height,
                source.width,
                source.nodata,
            )
        self._validate_grid(grid, mask_grid, self.mask_path)
        return LoadedFeatures(
            arrays,
            reference_dates,
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
                values = payload.get('acquisition_dates', payload.get('dates'))
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

    def _align_to_reference_dates(
        self,
        values: np.ndarray,
        source_dates: tuple[str, ...],
        reference_dates: tuple[str, ...],
        name: str,
    ) -> np.ndarray:
        """Align a static or temporal feature stack to model acquisition dates."""
        if values.shape[0] == 1:
            return np.broadcast_to(
                values,
                (len(reference_dates), *values.shape[1:]),
            ).copy()
        if len(source_dates) != values.shape[0]:
            raise ValueError(f'Feature {name} has incompatible dates and bands.')
        if source_dates == reference_dates:
            return values

        source_index = {value: index for index, value in enumerate(source_dates)}
        if len(source_index) != len(source_dates):
            raise ValueError(f'Feature {name} contains duplicate acquisition dates.')
        if self.temporal_alignment_method == 'exact':
            missing = [value for value in reference_dates if value not in source_index]
            if missing:
                raise ValueError(
                    f'Feature {name} is missing dates required by the model: '
                    f'{", ".join(missing[:3])}.',
                )
            indices = [source_index[value] for value in reference_dates]
            return values[indices]

        source_day_values = self._parse_dates(source_dates, name)
        reference_day_values = self._parse_dates(reference_dates, 'reference')
        indices = (
            np.searchsorted(source_day_values, reference_day_values, side='right') - 1
        )
        if np.any(indices < 0):
            first_missing = reference_dates[int(np.flatnonzero(indices < 0)[0])]
            raise ValueError(
                f'Feature {name} has no causal value on or before {first_missing}.',
            )
        return values[indices]

    @staticmethod
    def _parse_dates(values: tuple[str, ...], name: str) -> np.ndarray:
        """Convert ISO dates to sortable day numbers for causal as-of joining."""
        try:
            parsed = [date.fromisoformat(value).toordinal() for value in values]
        except ValueError as exc:
            raise ValueError(
                f'Feature {name} requires ISO dates for previous-date alignment.',
            ) from exc
        output = np.asarray(parsed, dtype=np.int64)
        if np.any(np.diff(output) <= 0):
            raise ValueError(f'Feature {name} dates must be strictly chronological.')
        return output
