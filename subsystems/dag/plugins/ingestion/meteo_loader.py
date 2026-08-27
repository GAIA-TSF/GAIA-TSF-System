"""Load dated meteorological GeoTIFFs into the common raster-series contract."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re

import numpy as np
import rasterio

from subsystems.dag.core.interfaces import RasterLoader
from subsystems.dag.utils.raster import RasterProfile, RasterTimeSeries


class MeteoRasterLoader(RasterLoader):
    """Load a dated, single-band meteorological raster series."""

    _date_pattern = re.compile(r'(\d{8})\.tif$')

    @property
    def name(self) -> str:
        """Return the plugin registry name."""
        return 'meteo_raster_loader'

    def load(self, directory: Path, filename_pattern: str) -> RasterTimeSeries:
        """Load matching single-band rasters in acquisition-date order.

        Filenames must contain YYYYMMDD. All rasters must share their spatial
        grid and nodata contract.
        """
        if not directory.exists():
            raise FileNotFoundError(
                f'Meteo input directory does not exist: {directory}'
            )
        paths = sorted(directory.glob(filename_pattern))
        if not paths:
            raise FileNotFoundError(
                f'No meteo rasters found in {directory} matching {filename_pattern}'
            )

        dated_paths = sorted((self._parse_date(path), path) for path in paths)
        dates = tuple(value for value, _ in dated_paths)
        if len(set(dates)) != len(dates):
            raise ValueError('Meteorological raster dates must be unique.')

        arrays: list[np.ndarray] = []
        profile: RasterProfile | None = None
        for _, path in dated_paths:
            with rasterio.open(path) as dataset:
                if dataset.count != 1:
                    raise ValueError(f'Meteo input raster must have one band: {path}')
                current = RasterProfile(
                    crs=dataset.crs,
                    transform=dataset.transform,
                    width=dataset.width,
                    height=dataset.height,
                    dtype=dataset.dtypes[0],
                    nodata=dataset.nodata,
                )
                if profile is None:
                    profile = current
                else:
                    self.validate_profile(profile, current, path)
                values = dataset.read(1).astype(np.float32)
                if dataset.nodata is not None:
                    values = np.where(values == dataset.nodata, np.nan, values)
                arrays.append(values)

        assert profile is not None
        return RasterTimeSeries(
            data=np.stack(arrays),
            dates=dates,
            profile=profile,
            source_paths=tuple(path for _, path in dated_paths),
        )

    def _parse_date(self, path: Path) -> date:
        match = self._date_pattern.search(path.name)
        if match is None:
            raise ValueError(
                'Invalid meteo filename. Expected a YYYYMMDD.tif suffix, '
                f'got {path.name}'
            )
        return datetime.strptime(match.group(1), '%Y%m%d').date()

    @staticmethod
    def validate_profile(
        expected: RasterProfile,
        current: RasterProfile,
        path: Path,
    ) -> None:
        """Raise ``ValueError`` when a candidate grid differs from a reference."""
        if current.crs != expected.crs:
            raise ValueError(
                f'CRS mismatch for {path}: {current.crs} != {expected.crs}'
            )
        if current.transform != expected.transform:
            raise ValueError(f'Resolution or transform mismatch for {path}.')
        if current.width != expected.width or current.height != expected.height:
            raise ValueError(f'Raster dimensions do not match for {path}.')
