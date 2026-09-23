"""Load and validate chronological Sentinel-1 LOS GeoTIFF stacks."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from subsystems.dag.core.interfaces import RasterLoader
from subsystems.dag.utils.raster import RasterProfile, RasterTimeSeries

LOGGER = logging.getLogger(__name__)


class Sentinel1LOSLoader(RasterLoader):
    """Load Sentinel-1 LOS deformation rasters."""

    _date_pattern = re.compile(r'tsf_los_(\d{8})\.tif$')

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return 'sentinel1_los_loader'

    def load(self, directory: Path, filename_pattern: str) -> RasterTimeSeries:
        """Load a chronologically sorted LOS raster time series.

        Args:
            directory: Directory containing LOS rasters.
            filename_pattern: Glob pattern for LOS rasters.

        Returns:
            Raster time series.

        Raises:
            FileNotFoundError: If the input directory or rasters are missing.
            ValueError: If filenames or raster metadata are inconsistent.
        """
        if not directory.exists():
            raise FileNotFoundError(f'LOS input directory does not exist: {directory}')

        paths = sorted(directory.glob(filename_pattern))
        if not paths:
            raise FileNotFoundError(
                f'No LOS rasters found in {directory} matching {filename_pattern}',
            )

        dated_paths = sorted((self._parse_date(path), path) for path in paths)
        dates = tuple(value for value, _ in dated_paths)
        self._validate_unique_dates(dates)

        arrays: list[np.ndarray] = []
        profile: RasterProfile | None = None
        for _, path in dated_paths:
            with rasterio.open(path) as dataset:
                current_profile = RasterProfile(
                    crs=dataset.crs,
                    transform=dataset.transform,
                    width=dataset.width,
                    height=dataset.height,
                    dtype=dataset.dtypes[0],
                    nodata=dataset.nodata,
                )
                if profile is None:
                    profile = current_profile
                else:
                    self._validate_profile(profile, current_profile, path)

                layer = dataset.read(1).astype(np.float32)
                if dataset.nodata is not None:
                    layer = np.where(layer == dataset.nodata, np.nan, layer)
                arrays.append(layer)

        if profile is None:
            raise ValueError('No raster metadata was loaded.')

        LOGGER.info('Loaded %s Sentinel-1 LOS rasters.', len(arrays))
        return RasterTimeSeries(
            data=np.stack(arrays),
            dates=dates,
            profile=profile,
            source_paths=tuple(path for _, path in dated_paths),
        )

    def _parse_date(self, path: Path) -> date:
        match = self._date_pattern.match(path.name)
        if not match:
            raise ValueError(
                f'Invalid LOS filename. Expected tsf_los_YYYYMMDD.tif, got {path.name}',
            )
        return date.fromisoformat(match.group(1))

    def _validate_unique_dates(self, dates: tuple[date, ...]) -> None:
        if len(set(dates)) != len(dates):
            raise ValueError('LOS raster acquisition dates must be unique.')

    def _validate_profile(
        self,
        expected: RasterProfile,
        current: RasterProfile,
        path: Path,
    ) -> None:
        if current.crs != expected.crs:
            raise ValueError(
                f'CRS mismatch for {path}: {current.crs} != {expected.crs}'
            )
        if current.transform != expected.transform:
            raise ValueError(f'Resolution or transform mismatch for {path}.')
        if current.width != expected.width or current.height != expected.height:
            raise ValueError(
                'Dimension mismatch for '
                f'{path}: {(current.height, current.width)} != '
                f'{(expected.height, expected.width)}',
            )
