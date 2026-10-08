from __future__ import annotations

import numpy as np


class TemporalWindowGenerator:
    """Generate rolling windows from temporal feature arrays."""

    def __init__(self, look_back: int = 1) -> None:
        self.look_back = max(1, look_back)

    def generate(self, values: np.ndarray) -> np.ndarray:
        """Generate lagged samples from a 2D array of shape (time, pixels)."""
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            values = values.reshape(values.shape[0], -1)

        samples: list[np.ndarray] = []
        for index in range(self.look_back - 1, values.shape[0]):
            window = values[index - self.look_back + 1 : index + 1, :]
            samples.append(window.reshape(-1))
        return np.vstack(samples) if samples else np.empty((0, values.shape[1] * self.look_back))
