from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import MonitoringPlugin
from core.registry import register_monitoring


@register_monitoring("zscore")
class ZScoreMonitoringPlugin(MonitoringPlugin):
    """Compute z-score based monitoring statistics."""

    name = "zscore"

    def evaluate(self, residuals: np.ndarray, config: Any) -> dict[str, Any]:
        residuals = np.asarray(residuals, dtype=float)
        zscore_config = getattr(getattr(config, "monitoring", None), "zscore", None)
        threshold = float(getattr(zscore_config, "threshold", 3.0))
        std = float(np.std(residuals))
        if std == 0.0:
            std = 1.0
        z_scores = (residuals - np.mean(residuals)) / std
        return {
            "z_scores": z_scores.tolist(),
            "threshold": threshold,
            "binary_anomaly": (np.abs(z_scores) >= threshold).tolist(),
        }
