"""Robust clean-water variability diagnostics for AMD acquisitions."""

import numpy as np
import warnings
from numpy.lib.stride_tricks import sliding_window_view

MAD_TO_SIGMA = 1.4826


def clean_water_variability(data, mask, threshold_sigma=3.0, minimum_valid_pixels=10):
    """Flag dates whose clean-water spatial variability is unusually high."""
    if threshold_sigma <= 0 or minimum_valid_pixels < 1:
        raise ValueError('Clean-water threshold must be positive and minimum pixels >= 1.')
    scales = np.full(data.shape[0], np.nan, dtype=float)
    counts = np.isfinite(data[:, mask]).sum(axis=1)
    for index in np.flatnonzero(counts >= minimum_valid_pixels):
        values = data[index, mask]
        values = values[np.isfinite(values)]
        median = np.median(values)
        scales[index] = MAD_TO_SIGMA * np.median(np.abs(values - median))
    finite = scales[np.isfinite(scales)]
    if not finite.size:
        raise ValueError('No acquisition has enough valid clean-water pixels.')
    centre = float(np.median(finite))
    spread = float(MAD_TO_SIGMA * np.median(np.abs(finite - centre)))
    threshold = centre + threshold_sigma * spread
    noisy = np.isfinite(scales) & (scales > threshold)
    return {
        'robust_sigma': scales,
        'valid_pixels': counts,
        'baseline_median_sigma': centre,
        'between_acquisition_robust_sigma': spread,
        'threshold': threshold,
        'noisy': noisy,
    }


def spatial_inconsistency(data, reference_mask, window_size=3, threshold_sigma=3.0,
                          minimum_valid_neighbors=3, minimum_reference_pixels=10):
    """Flag isolated pixels using local-median residuals and clean-water scale."""
    if (isinstance(window_size, bool) or not isinstance(window_size, int)
            or window_size < 3 or window_size % 2 == 0):
        raise ValueError('Spatial inconsistency window_size must be odd and >= 3.')
    if threshold_sigma <= 0 or minimum_valid_neighbors < 1:
        raise ValueError('Spatial threshold and minimum_valid_neighbors must be positive.')
    if minimum_valid_neighbors > window_size * window_size - 1:
        raise ValueError('minimum_valid_neighbors exceeds available neighboring pixels.')
    flags = np.zeros(data.shape, dtype=bool)
    thresholds = np.full(data.shape[0], np.nan)
    scales = np.full(data.shape[0], np.nan)
    radius = window_size // 2
    for index, image in enumerate(data):
        padded = np.pad(image, radius, constant_values=np.nan)
        windows = sliding_window_view(padded, (window_size, window_size)).copy()
        windows[..., radius, radius] = np.nan  # Compare against neighbors only.
        counts = np.isfinite(windows).sum(axis=(-2, -1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            local_median = np.nanmedian(windows, axis=(-2, -1))
        residual = image - local_median
        reference = residual[reference_mask & np.isfinite(residual)
                             & (counts >= minimum_valid_neighbors)]
        if reference.size < minimum_reference_pixels:
            continue
        centre = np.median(reference)
        scale = MAD_TO_SIGMA * np.median(np.abs(reference - centre))
        scales[index] = scale
        thresholds[index] = threshold_sigma * scale
        flags[index] = (np.isfinite(image) & (counts >= minimum_valid_neighbors)
                        & (np.abs(residual - centre) > thresholds[index]))
    return {'flags': flags, 'thresholds': thresholds, 'robust_sigma': scales}
