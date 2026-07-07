from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SplitIndices:
    """Indices for one train/validation/test split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class TemporalSplitter:
    """Create time-based splits for sequential data."""

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, n_samples: int) -> SplitIndices:
        """Split a sequence of samples into train/val/test sets."""
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")

        n_samples = max(3, int(n_samples))
        train_end = max(1, int(n_samples * self.train_ratio))
        val_end = max(train_end + 1, train_end + int(n_samples * self.val_ratio))
        test_end = max(val_end + 1, n_samples)

        return SplitIndices(
            train=np.arange(0, train_end),
            val=np.arange(train_end, val_end),
            test=np.arange(val_end, test_end),
        )
