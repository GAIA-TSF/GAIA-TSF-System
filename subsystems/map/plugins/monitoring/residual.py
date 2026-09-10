from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import MonitoringPlugin
from core.registry import register_monitoring


@register_monitoring("residual")
class ResidualMonitoringPlugin(MonitoringPlugin):
    """Compute basic residual statistics."""

    name = "residual"

    def evaluate(self, residuals: np.ndarray, config: Any) -> dict[str, Any]:
        residuals = np.asarray(residuals, dtype=float)
        residual_config = getattr(getattr(config, "monitoring", None), "residual", None)
        threshold = float(getattr(residual_config, "threshold", np.std(residuals) * 3.0))
        binary_anomaly = np.abs(residuals) >= threshold
        return {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "max_abs": float(np.max(np.abs(residuals))),
            "threshold": threshold,
            "binary_anomaly": binary_anomaly.tolist(),
        }
