from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import ExplainabilityPlugin
from core.registry import register_explainability


@register_explainability("shap")
class SHAPExplainabilityPlugin(ExplainabilityPlugin):
    """Placeholder SHAP explainability plugin."""

    name = "shap"

    def explain(self, model: Any, X: np.ndarray, config: Any) -> dict[str, Any]:  # noqa: N803
        return {"method": "shap", "shape": list(X.shape)}
