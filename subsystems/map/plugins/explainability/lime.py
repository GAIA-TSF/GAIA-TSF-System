from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import ExplainabilityPlugin
from core.registry import register_explainability


@register_explainability("lime")
class LIMEExplainabilityPlugin(ExplainabilityPlugin):
    """Placeholder LIME explainability plugin."""

    name = "lime"

    def explain(self, model: Any, X: np.ndarray, config: Any) -> dict[str, Any]:  # noqa: N803
        return {"method": "lime", "shape": list(X.shape)}
