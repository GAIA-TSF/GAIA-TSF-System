from __future__ import annotations

import logging
from typing import Any

from core.pipeline_executor import PipelineExecutor
from core.registry import OPERATION_REGISTRY

import pipelines.operations  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("map.learning_pipeline")


def run_learning(config: Any) -> dict[str, Any]:
    """Run the configured MAP learning DAG."""
    logger.info("Starting configured MAP learning pipeline")
    result = PipelineExecutor(config, OPERATION_REGISTRY).run("learning", initial_context={"input": None})
    if not isinstance(result, dict):
        raise TypeError("Configured learning pipeline must return a dictionary result.")
    return _json_ready(result)


def _json_ready(value: Any) -> Any:
    """Convert pipeline output into a lightweight public return value."""
    if isinstance(value, dict):
        return {
            key: _json_ready(item)
            for key, item in value.items()
            if key not in {"dataset", "model", "train_rows", "stable_mask"}
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
