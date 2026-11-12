"""Causal reference domain for preprocessing statistics (MAP contract 3.3)."""

from __future__ import annotations

import numpy as np


def reference_cells(shape: tuple[int, ...], reference: np.ndarray | None) -> np.ndarray:
    """Broadcast a time x space reference mask to a feature shape.

    ``None`` keeps the legacy behaviour (all cells), which uses future dates.
    """
    if reference is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(reference, dtype=bool)
    if len(shape) == 3 and mask.shape == shape[1:]:
        mask = mask[np.newaxis, :, :]
    try:
        return np.broadcast_to(mask, shape)
    except ValueError:
        raise ValueError(
            'Preprocessing reference mask cannot be broadcast to the feature shape.'
        ) from None
