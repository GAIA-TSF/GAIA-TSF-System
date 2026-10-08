"""Nodata-aware descriptive statistics for time series and feature rasters."""

from __future__ import annotations

from datetime import date

import numpy as np


def finite_values(data: np.ndarray) -> np.ndarray:
    """Return finite values from an array.

    Raises:
        ValueError: If the array contains no finite values.
    """
    values = data[np.isfinite(data)]
    if values.size == 0:
        raise ValueError('Dataset contains no finite values after masking.')
    return values


def median_absolute_deviation(values: np.ndarray) -> float:
    """Compute median absolute deviation."""
    median = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - median)))


def acquisition_statistics(values: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for one acquisition."""
    finite = finite_values(values)
    percentiles = np.nanpercentile(finite, [5, 25, 75, 95])
    return {
        'mean': float(np.nanmean(finite)),
        'median': float(np.nanmedian(finite)),
        'std': float(np.nanstd(finite)),
        'variance': float(np.nanvar(finite)),
        'minimum': float(np.nanmin(finite)),
        'maximum': float(np.nanmax(finite)),
        'mad': median_absolute_deviation(finite),
        'percentiles': {
            'p5': float(percentiles[0]),
            'p25': float(percentiles[1]),
            'p75': float(percentiles[2]),
            'p95': float(percentiles[3]),
        },
    }


def time_series_statistics(
    data: np.ndarray,
    dates: tuple[date, ...],
    histogram_bins: int,
) -> dict[str, object]:
    """Compute nested descriptive statistics for a raster time series."""
    if data.shape[0] != len(dates):
        raise ValueError('Number of raster layers does not match dates.')

    global_values = finite_values(data)
    hist_counts, hist_edges = np.histogram(global_values, bins=histogram_bins)

    return {
        'per_acquisition': {
            acquisition_date.isoformat(): acquisition_statistics(data[index])
            for index, acquisition_date in enumerate(dates)
        },
        'overall': {
            'global_min': float(np.nanmin(global_values)),
            'global_max': float(np.nanmax(global_values)),
            'overall_mean': float(np.nanmean(global_values)),
            'overall_std': float(np.nanstd(global_values)),
            'global_histogram': {
                'bins': hist_edges.tolist(),
                'counts': hist_counts.astype(int).tolist(),
            },
        },
    }


def feature_statistics(values: np.ndarray) -> dict[str, float | int | None]:
    """Compute compact metadata statistics for a feature raster."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            'valid_pixels': 0,
            'minimum': None,
            'maximum': None,
            'mean': None,
            'std': None,
        }

    return {
        'valid_pixels': int(finite.size),
        'minimum': float(np.nanmin(finite)),
        'maximum': float(np.nanmax(finite)),
        'mean': float(np.nanmean(finite)),
        'std': float(np.nanstd(finite)),
    }
