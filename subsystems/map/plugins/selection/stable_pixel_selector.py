from __future__ import annotations

from typing import Any

import numpy as np


class StablePixelSelector:
    """Select stable pixels suitable for baseline model training."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.threshold = float(getattr(config, "stable_pixel_std_threshold", 0.008))

    def select(self, feature_stack: np.ndarray) -> np.ndarray:
        """Return a boolean mask for pixels whose temporal standard deviation is low enough."""
        values = np.asarray(feature_stack, dtype=float)
        if values.ndim == 1:
            return np.std(values) < self.threshold
        if values.ndim not in {2, 3}:
            raise ValueError("Expected values shaped as (time, pixels) or (time, height, width).")

        std_values = np.std(values, axis=0)
        return std_values < self.threshold
