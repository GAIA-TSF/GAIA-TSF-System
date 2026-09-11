"""Configurable two-band AMD indicator calculation."""

import numpy as np


def calculate_amd_index(first, second, method, denominator_epsilon=0.0):
    """Calculate first/second or first-second, preserving missing observations."""
    if method not in {'ratio', 'difference'}:
        raise ValueError('AMD method must be ratio or difference.')
    if not np.isfinite(denominator_epsilon) or denominator_epsilon < 0:
        raise ValueError('denominator_epsilon must be finite and nonnegative.')
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError('AMD input bands must have identical shapes.')
    valid = np.isfinite(first) & np.isfinite(second)
    output = np.full(first.shape, np.nan)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if method == 'ratio':
            valid &= np.abs(second) > denominator_epsilon
            np.divide(first, second, out=output, where=valid)
        else:
            np.subtract(first, second, out=output, where=valid)
    return np.where(np.isfinite(output), output, np.nan)
