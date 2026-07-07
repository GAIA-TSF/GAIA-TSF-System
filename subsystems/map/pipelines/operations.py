from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.experiment_manager import ExperimentManager
from core.interfaces import Dataset
from core.model_registry import ModelRegistry
from core.registry import MODEL_REGISTRY, VARIABLE_REGISTRY, register_operation
from dataset.dataset_builder import DatasetBuilder
from pipelines.monitoring_pipeline import run_monitoring_pipeline
from plugins.selection.stable_pixel_selector import StablePixelSelector

import plugins.models.gbr  # noqa: F401
import plugins.models.lstm  # noqa: F401
import plugins.models.rf  # noqa: F401
import plugins.models.xgb  # noqa: F401
import plugins.variables.amd  # noqa: F401
import plugins.variables.slope  # noqa: F401

logger = logging.getLogger("map.pipeline_operations")


@register_operation("tensor_loader")
def tensor_loader(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> Dataset:
    """Build the configured MAP dataset from engineered feature tensors."""
    return DatasetBuilder(config).build()


@register_operation("splitter")
def splitter(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> Any:
    """Return the incoming dataset after configured temporal splitting."""
    return _first_input(inputs)


@register_operation("windowing")
def windowing(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> Any:
    """Return the incoming dataset after configured temporal windowing."""
    return _first_input(inputs)


@register_operation("feature_engineering")
def feature_engineering(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> Any:
    """Return engineered feature samples for downstream MAP nodes."""
    return _first_input(inputs)


@register_operation("stable_pixel_selection")
def stable_pixel_selection(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Select stable pixels and mark training rows that may be used by models."""
    dataset = _as_dataset(_first_input(inputs))
    if dataset.stable_selection_values is None:
        raise ValueError("Dataset does not include values for stable pixel selection.")

    selector_config = (
        getattr(config, "stable_pixel_selector", None)
        or getattr(config, "stable_pixel", None)
        or type("Cfg", (), {"stable_pixel_std_threshold": 0.008})()
    )
    stable_mask = StablePixelSelector(selector_config).select(dataset.stable_selection_values)
    stable_mask_flat = np.asarray(stable_mask, dtype=bool).reshape(-1)
    if dataset.train_pixel_indices.size:
        train_rows = stable_mask_flat[dataset.train_pixel_indices]
    else:
        train_rows = np.ones(dataset.X_train.shape[0], dtype=bool)
    if not np.any(train_rows):
        raise ValueError("Stable pixel selection produced no training samples.")

    logger.info("Selected %s stable pixels for MAP baseline training", int(np.count_nonzero(stable_mask_flat)))
    return {"dataset": dataset, "stable_mask": stable_mask_flat, "train_rows": train_rows}


@register_operation("trainer")
def trainer(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Train and persist the configured predictive model."""
    training_input = _first_input(inputs)
    dataset = _as_dataset(training_input)
    train_rows = _train_rows(training_input, dataset)
    stable_mask = np.asarray(training_input.get("stable_mask", []), dtype=bool) if isinstance(training_input, dict) else None

    variable_name = _active_variable(config)
    variable_plugin = VARIABLE_REGISTRY.get(variable_name)
    if variable_plugin is None:
        raise KeyError(f"Unknown variable plugin '{variable_name}'.")

    model_name = _active_model(config)
    allowed_models = variable_plugin.allowed_models()
    if allowed_models and model_name not in allowed_models:
        raise ValueError(f"Model '{model_name}' is not allowed for variable '{variable_name}'.")

    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise KeyError(f"Unknown model plugin '{model_name}'.")

    model_config = getattr(getattr(config, "models", None), model_name, None)
    model = model_cls(model_config or type("Cfg", (), {"n_estimators": 200, "random_state": 42})())
    model.train(dataset.X_train[train_rows], dataset.y_train[train_rows])

    output_dir = Path("results/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{variable_name}_{model_name}.joblib"
    model.save(model_path)

    return {
        "dataset": dataset,
        "model": model,
        "model_path": str(model_path),
        "model_name": model_name,
        "variable_name": variable_name,
        "train_rows": train_rows,
        "stable_mask": stable_mask,
    }


@register_operation("validation")
def validation(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Validate a trained model and register training artifacts."""
    training_state = dict(_first_input(inputs))
    dataset = _as_dataset(training_state)
    model = training_state["model"]

    predictions = np.asarray(model.predict(dataset.X_val), dtype=float)
    y_val = np.asarray(dataset.y_val, dtype=float)
    rmse = float(np.sqrt(np.mean((predictions - y_val) ** 2)))
    mae = float(np.mean(np.abs(predictions - y_val)))
    stable_mask = training_state.get("stable_mask")
    train_rows = np.asarray(training_state.get("train_rows", []), dtype=bool)
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "stable_pixel_count": int(np.count_nonzero(stable_mask)) if stable_mask is not None else 0,
        "stable_training_sample_count": int(np.count_nonzero(train_rows)),
    }

    variable_name = training_state["variable_name"]
    model_name = training_state["model_name"]
    output_dir = Path("results/models")
    metrics_path = output_dir / f"{variable_name}_{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    feature_pipeline = getattr(config, "active_feature_pipeline", None) or getattr(config, "feature_pipeline", "temporal")
    ModelRegistry(output_dir).register(
        f"{variable_name}_{model_name}",
        {
            "variable": variable_name,
            "model_type": model_name,
            "feature_pipeline": feature_pipeline,
            "features": dataset.feature_names,
            "training_configuration": _to_plain_dict(getattr(config, "training", {})),
            "pipeline_configuration": _to_plain_dict(getattr(getattr(config, "pipelines", None), "learning", {})),
            "training_metrics": metrics,
            "model_path": training_state["model_path"],
        },
    )

    ExperimentManager(Path("results/experiments")).register(
        getattr(getattr(config, "experiment", None), "name", "map_experiment"),
        {"variable": variable_name, "model": model_name, "metrics": metrics},
    )

    training_state["metrics"] = metrics
    return {"model_path": training_state["model_path"], "metrics": metrics, "dataset": dataset}


@register_operation("predictor")
def predictor(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Load the configured model and persist prediction/residual outputs."""
    dataset = _as_dataset(_first_input(inputs))
    variable_name = _active_variable(config)
    model_name = _active_model(config)
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise KeyError(f"Unknown model plugin '{model_name}'.")

    model_path = Path(
        getattr(getattr(config, "inference", None), "model_path", "")
        or Path("results/models") / f"{variable_name}_{model_name}.joblib"
    )
    model = model_cls.load(model_path)
    predictions = np.asarray(model.predict(dataset.X_test), dtype=float)
    residuals = np.asarray(dataset.y_test, dtype=float) - predictions

    prediction_dir = Path("results/predictions")
    residual_dir = Path("results/residuals")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"prediction_{variable_name}.npy"
    residual_path = residual_dir / f"residual_{variable_name}.npy"
    np.save(prediction_path, predictions)
    np.save(residual_path, residuals)
    (residual_dir / f"residual_statistics_{variable_name}.json").write_text(
        json.dumps({"mean": float(np.mean(residuals)), "std": float(np.std(residuals))}, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset": dataset,
        "model": model,
        "model_path": str(model_path),
        "predictions": predictions,
        "residuals": residuals,
        "prediction_path": str(prediction_path),
        "residual_path": str(residual_path),
    }


@register_operation("residual_analysis")
def residual_analysis(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Compute residual statistics from a prediction result."""
    prediction_state = dict(_first_input(inputs))
    residuals = np.asarray(prediction_state["residuals"], dtype=float)
    prediction_state["residual_statistics"] = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
    }
    return prediction_state


@register_operation("trend_detection")
def trend_detection(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Summarize temporal trend in prediction output."""
    prediction_state = dict(_first_input(inputs))
    predictions = np.asarray(prediction_state["predictions"], dtype=float)
    deltas = np.diff(predictions) if predictions.size > 1 else np.asarray([], dtype=float)
    prediction_state["trend"] = {
        "mean_delta": float(np.mean(deltas)) if deltas.size else 0.0,
        "max_delta": float(np.max(deltas)) if deltas.size else 0.0,
        "min_delta": float(np.min(deltas)) if deltas.size else 0.0,
    }
    return prediction_state


@register_operation("anomaly_detection")
def anomaly_detection(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Run configured monitoring plugins as anomaly detectors."""
    prediction_state = dict(_first_input(inputs))
    residuals = np.asarray(prediction_state["residuals"], dtype=float)
    prediction_state["anomaly"] = run_monitoring_pipeline(config, residuals)
    return prediction_state


@register_operation("risk_scoring")
def risk_scoring(config: Any, inputs: list[Any], context: dict[str, Any], node_config: Any) -> dict[str, Any]:
    """Combine trend and anomaly outputs into a lightweight risk score."""
    state: dict[str, Any] = {}
    for item in inputs:
        if isinstance(item, dict):
            state.update(item)

    anomaly = state.get("anomaly", {})
    combined = anomaly.get("combined", {}) if isinstance(anomaly, dict) else {}
    trend = state.get("trend", {})
    anomaly_score = float(combined.get("anomaly_score_max", 0.0))
    trend_score = abs(float(trend.get("mean_delta", 0.0))) if isinstance(trend, dict) else 0.0
    state["risk"] = {
        "risk_score": anomaly_score + trend_score,
        "anomaly_score": anomaly_score,
        "trend_score": trend_score,
    }
    return state


def _first_input(inputs: list[Any]) -> Any:
    """Return the first non-None input value."""
    for item in inputs:
        if item is not None:
            return item
    return None


def _as_dataset(value: Any) -> Dataset:
    """Extract a Dataset from a direct value or operation state dictionary."""
    if isinstance(value, Dataset):
        return value
    if isinstance(value, dict) and isinstance(value.get("dataset"), Dataset):
        return value["dataset"]
    raise TypeError("Expected a Dataset or operation state containing a Dataset.")


def _train_rows(value: Any, dataset: Dataset) -> np.ndarray:
    """Extract stable training rows or default to all training rows."""
    if isinstance(value, dict) and "train_rows" in value:
        return np.asarray(value["train_rows"], dtype=bool)
    return np.ones(dataset.X_train.shape[0], dtype=bool)


def _active_variable(config: Any) -> str:
    """Return the active variable name from config."""
    return str(getattr(config, "active_variable", None) or getattr(config, "variable", "slope"))


def _active_model(config: Any) -> str:
    """Return the active model name from config."""
    return str(getattr(config, "active_model", None) or getattr(config, "model", "rf"))


def _to_plain_dict(value: Any) -> Any:
    """Convert config wrapper objects into JSON-serializable containers."""
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _to_plain_dict(item) for key, item in vars(value).items()}
    return value
