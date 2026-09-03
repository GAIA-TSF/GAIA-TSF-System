from __future__ import annotations

from datetime import date

import numpy as np


def days_since_start(dates: tuple[date, ...]) -> np.ndarray:
    """Convert acquisition dates to day offsets from the first date."""
    if len(dates) == 0:
        raise ValueError('At least one acquisition date is required.')
    first_date = dates[0]
    return np.array([(value - first_date).days for value in dates], dtype=np.float64)


def validate_temporal_axis(data: np.ndarray, dates: tuple[date, ...]) -> None:
    """Validate temporal stack and dates."""
    if data.ndim != 3:
        raise ValueError('Temporal raster stack must have shape (time, rows, cols).')
    if data.shape[0] != len(dates):
        raise ValueError('Number of raster layers does not match acquisition dates.')
    if len(dates) != len(set(dates)):
        raise ValueError('Acquisition dates must be unique.')
    if len(dates) > 1 and np.any(np.diff(days_since_start(dates)) <= 0):
        raise ValueError('Acquisition dates must be strictly chronological.')


def temporal_gradient(
    data: np.ndarray, dates: tuple[date, ...], order: int
) -> np.ndarray:
    """Compute a temporal derivative using acquisition day spacing.

    Args:
        data: Raster stack with shape ``(time, rows, cols)``.
        dates: Chronological acquisition dates.
        order: Derivative order.

    Returns:
        Derivative stack with the same shape as ``data``.

    Raises:
        ValueError: If there are not enough acquisitions for the derivative.
    """
    validate_temporal_axis(data, dates)
    if order < 1:
        raise ValueError('Derivative order must be at least 1.')
    if len(dates) < order + 1:
        raise ValueError(
            f'At least {order + 1} acquisitions are required for derivative order '
            f'{order}.',
        )

    offsets = days_since_start(dates)
    gradient = data.astype(np.float64)
    for _ in range(order):
        gradient = np.gradient(gradient, offsets, axis=0, edge_order=1)
    return gradient.astype(np.float32)


def nanmean_time(data: np.ndarray) -> np.ndarray:
    """Compute a warning-free mean over time."""
    counts = np.sum(np.isfinite(data), axis=0)
    sums = np.nansum(data, axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(data.shape[1:], np.nan, dtype=np.float32),
        where=counts > 0,
    )


def linear_trend(data: np.ndarray, dates: tuple[date, ...]) -> np.ndarray:
    """Compute per-pixel linear trend slope with least squares."""
    validate_temporal_axis(data, dates)
    if len(dates) < 2:
        raise ValueError('At least two acquisitions are required for trend.')

    x = days_since_start(dates)
    valid = np.isfinite(data)
    counts = np.sum(valid, axis=0)
    y = np.where(valid, data, 0.0)
    x_3d = x[:, np.newaxis, np.newaxis]

    sum_x = np.sum(np.where(valid, x_3d, 0.0), axis=0)
    sum_y = np.sum(y, axis=0)
    sum_xy = np.sum(np.where(valid, x_3d * data, 0.0), axis=0)
    sum_x2 = np.sum(np.where(valid, x_3d * x_3d, 0.0), axis=0)
    denominator = counts * sum_x2 - sum_x * sum_x

    return np.divide(
        counts * sum_xy - sum_x * sum_y,
        denominator,
        out=np.full(data.shape[1:], np.nan, dtype=np.float32),
        where=(counts >= 2) & (denominator != 0),
    ).astype(np.float32)
