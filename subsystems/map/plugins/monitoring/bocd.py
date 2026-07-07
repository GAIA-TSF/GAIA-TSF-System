from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import MonitoringPlugin
from core.registry import register_monitoring


@register_monitoring("bocd")
class BOCDMonitoringPlugin(MonitoringPlugin):
    """A simple placeholder Bayesian online change-point detector."""

    name = "bocd"

    def evaluate(self, residuals: np.ndarray, config: Any) -> dict[str, Any]:
        residuals = np.asarray(residuals, dtype=float)
        return {"change_points": np.where(np.abs(residuals) > np.std(residuals))[0].tolist()}
