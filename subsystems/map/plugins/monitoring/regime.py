from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import MonitoringPlugin
from core.registry import register_monitoring


@register_monitoring("regime")
class RegimeMonitoringPlugin(MonitoringPlugin):
    """Classify residual regimes into acceleration/deceleration/steady."""

    name = "regime"

    def evaluate(self, residuals: np.ndarray, config: Any) -> dict[str, Any]:
        residuals = np.asarray(residuals, dtype=float)
        if residuals.size < 2:
            return {"regime": "steady"}
        diffs = np.diff(residuals)
        positive = np.mean(diffs > 0)
        negative = np.mean(diffs < 0)
        if positive > negative:
            regime = "acceleration"
        elif negative > positive:
            regime = "deceleration"
        else:
            regime = "steady"
        return {"regime": regime}
