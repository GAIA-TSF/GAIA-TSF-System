from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.registry import EXPLAINABILITY_REGISTRY
import plugins.explainability.dice  # noqa: F401
import plugins.explainability.lime  # noqa: F401
import plugins.explainability.shap  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("map.explainability_pipeline")


def run_explainability_pipeline(config: Any, model: Any, X: np.ndarray) -> dict[str, Any]:  # noqa: N803
    """Run the registered explainability plugins on a trained model and features."""
    methods = list(getattr(getattr(config, "explainability", None), "methods", []) or [])
    if not methods:
        methods = ["shap"]

    output: dict[str, Any] = {}
    output_dir = Path("results/explainability")
    output_dir.mkdir(parents=True, exist_ok=True)
    for method_name in methods:
        plugin_cls = EXPLAINABILITY_REGISTRY.get(method_name)
        if plugin_cls is None:
            logger.warning("Explainability plugin '%s' is not registered.", method_name)
            continue
        plugin = plugin_cls()
        output[method_name] = plugin.explain(model, X, config)
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "summary.json").write_text(json.dumps(output[method_name], indent=2), encoding="utf-8")

    return output
