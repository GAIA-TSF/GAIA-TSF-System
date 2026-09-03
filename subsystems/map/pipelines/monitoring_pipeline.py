from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.registry import MONITORING_REGISTRY
import plugins.monitoring.bocd  # noqa: F401
import plugins.monitoring.cusum  # noqa: F401
import plugins.monitoring.regime  # noqa: F401
import plugins.monitoring.residual  # noqa: F401
import plugins.monitoring.zscore  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("map.monitoring_pipeline")


def run_monitoring_pipeline(config: Any, residuals: np.ndarray) -> dict[str, Any]:
    """Run the registered monitoring plugins on residuals."""
    residuals = np.asarray(residuals, dtype=float)
    methods = list(getattr(getattr(config, "monitoring", None), "methods", []) or [])
    if not methods:
        methods = ["residual"]

    output: dict[str, Any] = {}
    score_components: list[np.ndarray] = []
    for method_name in methods:
        plugin_cls = MONITORING_REGISTRY.get(method_name)
        if plugin_cls is None:
            logger.warning("Monitoring plugin '%s' is not registered.", method_name)
            continue
        plugin = plugin_cls()
        output[method_name] = plugin.evaluate(residuals, config)
        score_components.append(_score_from_plugin_output(output[method_name], residuals.shape))

    combined_score = (
        np.mean(np.vstack([component.reshape(1, -1) for component in score_components]), axis=0)
        if score_components
        else np.zeros(residuals.size, dtype=float)
    ).reshape(residuals.shape)
    binary_threshold = float(getattr(getattr(config, "monitoring", None), "anomaly_threshold", 1.0))
    anomaly_binary = combined_score >= binary_threshold
    output["combined"] = {
        "anomaly_score_mean": float(np.mean(combined_score)),
        "anomaly_score_max": float(np.max(combined_score)) if combined_score.size else 0.0,
        "anomaly_count": int(np.count_nonzero(anomaly_binary)),
        "threshold": binary_threshold,
    }

    output_dir = Path("results/anomalies")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "anomaly_score.npy", combined_score)
    np.save(output_dir / "anomaly_binary.npy", anomaly_binary)
    (output_dir / "anomaly_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _score_from_plugin_output(plugin_output: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    """Convert a plugin output dictionary into a normalized score vector."""
    size = int(np.prod(shape))
    if "z_scores" in plugin_output:
        return np.abs(np.asarray(plugin_output["z_scores"], dtype=float)).reshape(-1)
    if "binary_anomaly" in plugin_output:
        return np.asarray(plugin_output["binary_anomaly"], dtype=float).reshape(-1)
    if "cumulative_abs_residual" in plugin_output:
        values = np.asarray(plugin_output["cumulative_abs_residual"], dtype=float).reshape(-1)
        maximum = float(np.max(values)) if values.size else 0.0
        return values / maximum if maximum else np.zeros(size, dtype=float)
    if "change_points" in plugin_output:
        scores = np.zeros(size, dtype=float)
        indices = np.asarray(plugin_output["change_points"], dtype=int)
        scores[indices[(indices >= 0) & (indices < size)]] = 1.0
        return scores
    return np.zeros(size, dtype=float)
