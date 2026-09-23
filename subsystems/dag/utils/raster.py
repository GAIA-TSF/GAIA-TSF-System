"""Shared raster data contracts, masking operations, and GeoTIFF writers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds


@dataclass(frozen=True)
class RasterProfile:
    """Spatial metadata shared by a raster time series."""

    crs: object
    transform: object
    width: int
    height: int
    dtype: str
    nodata: float | int | None


@dataclass(frozen=True)
class RasterTimeSeries:
    """Chronologically ordered raster stack."""

    data: np.ndarray
    dates: tuple[date, ...]
    profile: RasterProfile
    source_paths: tuple[Path, ...]


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a TSF mask to a time-series stack.

    Args:
        data: Array with shape ``(time, rows, cols)``.
        mask: Boolean or numeric mask with shape ``(rows, cols)``.

    Returns:
        Data with values outside the mask set to ``NaN``.

    Raises:
        ValueError: If the mask shape does not match the raster stack.
    """
    if data.ndim != 3:
        raise ValueError('LOS raster stack must have shape (time, rows, cols).')
    if mask.shape != data.shape[1:]:
        raise ValueError(
            'TSF mask dimensions do not match LOS raster dimensions: '
            f'{mask.shape} != {data.shape[1:]}',
        )

    mask_bool = mask.astype(bool)
    return np.where(mask_bool[np.newaxis, :, :], data, np.nan)


def temporal_mean(data: np.ndarray) -> np.ndarray:
    """Compute pixel-wise temporal mean."""
    counts = np.sum(np.isfinite(data), axis=0)
    sums = np.nansum(data, axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(data.shape[1:], np.nan, dtype=np.float32),
        where=counts > 0,
    )


def temporal_std(data: np.ndarray) -> np.ndarray:
    """Compute pixel-wise temporal standard deviation."""
    counts = np.sum(np.isfinite(data), axis=0)
    means = temporal_mean(data)
    squared_deviation = np.where(
        np.isfinite(data),
        np.square(data - means[np.newaxis, :, :]),
        0.0,
    )
    variance = np.divide(
        np.sum(squared_deviation, axis=0),
        counts,
        out=np.full(data.shape[1:], np.nan, dtype=np.float32),
        where=counts > 0,
    )
    return np.sqrt(variance)


def write_single_band_raster(
    path: Path,
    values: np.ndarray,
    profile: RasterProfile,
    driver: str,
    band_name: str | None = None,
) -> None:
    """Write a single-band raster using the source spatial profile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = profile.nodata
    output_values = values.astype(np.float32)
    if nodata is not None and np.isfinite(nodata):
        output_values = np.where(np.isfinite(output_values), output_values, nodata)

    with rasterio.open(
        path,
        'w',
        driver=driver,
        height=profile.height,
        width=profile.width,
        count=1,
        dtype='float32',
        crs=profile.crs,
        transform=profile.transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(output_values.astype(np.float32), 1)
        if band_name is not None:
            dataset.set_band_description(1, band_name)
    write_raster_sidecar(path, profile, (band_name,) if band_name else None)


def write_raster(
    path: Path,
    values: np.ndarray,
    profile: RasterProfile,
    driver: str,
    band_names: tuple[str, ...] | None = None,
) -> None:
    """Write a 2D raster or 3D temporal stack using the source spatial profile."""
    if values.ndim == 2:
        band_name = band_names[0] if band_names else None
        write_single_band_raster(path, values, profile, driver, band_name)
        return
    if values.ndim != 3:
        raise ValueError(
            'Feature rasters must have shape (rows, cols) or (time, rows, cols).',
        )
    if values.shape[1:] != (profile.height, profile.width):
        raise ValueError(
            'Feature raster dimensions do not match profile dimensions: '
            f'{values.shape[1:]} != {(profile.height, profile.width)}',
        )
    if band_names is not None and len(band_names) != values.shape[0]:
        raise ValueError(
            'Number of band names must match raster band count: '
            f'{len(band_names)} != {values.shape[0]}',
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = profile.nodata
    output_values = values.astype(np.float32)
    if nodata is not None and np.isfinite(nodata):
        output_values = np.where(np.isfinite(output_values), output_values, nodata)

    with rasterio.open(
        path,
        'w',
        driver=driver,
        height=profile.height,
        width=profile.width,
        count=output_values.shape[0],
        dtype='float32',
        crs=profile.crs,
        transform=profile.transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(output_values.astype(np.float32))
        if band_names is not None:
            for band_index, band_name in enumerate(band_names, start=1):
                dataset.set_band_description(band_index, band_name)
    write_raster_sidecar(path, profile, band_names)


def write_raster_sidecar(
    path: Path,
    profile: RasterProfile,
    band_names: tuple[str | None, ...] | None = None,
    *,
    properties: dict[str, Any] | None = None,
) -> Path:
    """Write a STAC-style JSON sidecar for an exported raster.

    The sidecar shares the raster stem, for example ``velocity.tif`` and
    ``velocity.json``.  This is intentionally compatible with the supplied
    Sentinel-2 item convention while remaining useful for generic derived
    GeoTIFFs that do not have sensor-specific metadata.
    """
    west, south, east, north = array_bounds(
        profile.height,
        profile.width,
        profile.transform,
    )
    crs = rasterio.crs.CRS.from_user_input(profile.crs)
    if crs.is_geographic:
        bbox = [west, south, east, north]
    else:
        bbox = list(transform_bounds(crs, 'EPSG:4326', west, south, east, north))
    geometry = {
        'type': 'Polygon',
        'coordinates': [[
            [bbox[0], bbox[1]], [bbox[0], bbox[3]],
            [bbox[2], bbox[3]], [bbox[2], bbox[1]], [bbox[0], bbox[1]],
        ]],
    }
    names = tuple(name for name in (band_names or ()) if name)
    item_properties: dict[str, Any] = {
        'proj:epsg': crs.to_epsg(),
        'proj:shape': [profile.height, profile.width],
        'proj:transform': list(profile.transform)[:6],
        'raster:bands': [
            {
                'name': name or f'band_{index}',
                'data_type': str(profile.dtype),
                'nodata': _json_number(profile.nodata),
            }
            for index, name in enumerate(band_names or (None,), start=1)
        ],
        'gaia:product_type': path.stem,
    }
    if names and all(_is_iso_date(value) for value in names):
        datetimes = [f'{value}T00:00:00Z' for value in names]
        item_properties['start_datetime'] = datetimes[0]
        item_properties['end_datetime'] = datetimes[-1]
        if len(datetimes) == 1:
            item_properties['datetime'] = datetimes[0]
    if properties:
        item_properties.update(properties)
    payload = {
        'type': 'Feature',
        'stac_version': '1.0.0',
        'id': path.stem,
        'bbox': bbox,
        'geometry': geometry,
        'properties': item_properties,
        'assets': {
            'data': {
                'href': f'./{path.name}',
                'type': 'image/tiff; application=geotiff',
                'roles': ['data'],
            },
        },
        'collection': 'gaia-tsf-derived',
    }
    sidecar = path.with_suffix('.json')
    sidecar.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return sidecar


def _is_iso_date(value: str) -> bool:
    """Return whether a band label is an ISO calendar date."""
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _json_number(value: float | int | None) -> float | int | None:
    """Convert raster numeric metadata to strict JSON-compatible values."""
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else numeric
