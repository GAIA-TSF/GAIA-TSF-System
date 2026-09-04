"""Backward-compatible monitoring entry point."""

from __future__ import annotations

import numpy as np


def run_monitoring(
    residuals: np.ndarray, config: dict[str, object]
) -> dict[str, object]:
    """Deprecated adapter retained for callers of the prototype API."""
    return {
        'residual_count': int(np.count_nonzero(np.isfinite(residuals))),
        'config': config,
    }
