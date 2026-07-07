from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import ExplainabilityPlugin
from core.registry import register_explainability


@register_explainability("dice")
class DICEExplainabilityPlugin(ExplainabilityPlugin):
    """Placeholder DiCE explainability plugin."""

    name = "dice"

    def explain(self, model: Any, X: np.ndarray, config: Any) -> dict[str, Any]:  # noqa: N803
        return {"method": "dice", "shape": list(X.shape)}
