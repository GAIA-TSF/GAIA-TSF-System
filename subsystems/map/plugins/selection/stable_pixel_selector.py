"""Reference stable-pixel selector for baseline training."""

from __future__ import annotations

import numpy as np


class StablePixelSelector:
    """Select pixels whose temporal target standard deviation is sufficiently low."""

    def __init__(self, stable_pixel_std_threshold: float) -> None:
        if stable_pixel_std_threshold <= 0:
            raise ValueError("stable_pixel_std_threshold must be positive.")
        self.stable_pixel_std_threshold = stable_pixel_std_threshold

    def select(self, observations: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Return a stable spatial mask.

        Args:
            observations: Target stack ordered as ``(time, rows, columns)``.
            valid_mask: TSF spatial mask shaped ``(rows, columns)``.

        Returns:
            Boolean stable-pixel mask.
        """
        if observations.ndim != 3 or observations.shape[1:] != valid_mask.shape:
            raise ValueError("Observations and TSF mask have incompatible dimensions.")
        finite_count = np.sum(np.isfinite(observations), axis=0)
        temporal_std = np.nanstd(observations, axis=0)
        stable = (
            valid_mask.astype(bool)
            & (finite_count >= 2)
            & np.isfinite(temporal_std)
            & (temporal_std < self.stable_pixel_std_threshold)
        )
        if not np.any(stable):
            raise ValueError("No stable pixels satisfy the configured standard deviation threshold.")
        return stable
