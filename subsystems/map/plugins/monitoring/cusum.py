from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import MonitoringPlugin
from core.registry import register_monitoring


@register_monitoring("cusum")
class CUSUMMonitoringPlugin(MonitoringPlugin):
    """A minimal CUSUM monitoring implementation."""

    name = "cusum"

    def evaluate(self, residuals: np.ndarray, config: Any) -> dict[str, Any]:
        residuals = np.asarray(residuals, dtype=float)
        cusum_config = getattr(getattr(config, "monitoring", None), "cusum", None)
        threshold = float(getattr(cusum_config, "threshold", 5.0))
        cumulative = np.cumsum(np.abs(residuals))
        return {
            "cumulative_abs_residual": cumulative.tolist(),
            "threshold": threshold,
            "binary_anomaly": (cumulative >= threshold).tolist(),
        }
