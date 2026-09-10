from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.pipeline_executor import PipelineExecutor
from core.registry import OPERATION_REGISTRY

import pipelines.operations  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("map.inference_pipeline")


def run_inference(config: Any) -> dict[str, Any]:
    """Run the configured MAP inference DAG."""
    logger.info("Starting configured MAP inference pipeline")
    result = PipelineExecutor(config, OPERATION_REGISTRY).run("inference", initial_context={"input": None})
    if not isinstance(result, dict):
        raise TypeError("Configured inference pipeline must return a dictionary result.")
    return _json_ready(result)


def _json_ready(value: Any) -> Any:
    """Convert NumPy-heavy operation output into a public dictionary result."""
    if isinstance(value, dict):
        return {
            key: _json_ready(item)
            for key, item in value.items()
            if key not in {"dataset", "model"}
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
