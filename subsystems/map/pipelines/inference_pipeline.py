"""Scenario 1 inference, residual analysis and anomaly detection workflow."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import DatasetBuilder, FeatureLoader
from subsystems.map.monitoring import ResidualAnalyzer, StatisticalAnomalyDetector
from subsystems.map.utils.artifacts import write_diagnostics


LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    """Predict every valid TSF pixel, then derive residual and anomaly products."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config_path = Path(str(config["_config_path"]))

    def run(self) -> dict[str, Any]:
        """Run inference across the configured TSF mask and persist all products."""
        import subsystems.map.plugins.models  # noqa: F401

        dataset_config = self._named_config("datasets", self._name("dataset"))
        feature_names = [str(value) for value in dataset_config["features"]]
        target_feature = str(dataset_config["target_feature"])
        loaded = FeatureLoader(self._feature_paths(), self._path(dataset_config["mask_path"])).load(
            list(dict.fromkeys([*feature_names, target_feature])),
        )
        dataset = DatasetBuilder().build(loaded, feature_names, target_feature)
        model_name = self._name("model")
        model_path = self._path(self.config["outputs"]["root"]) / "models" / "baseline_model.pkl"
        model = MODEL_REGISTRY[model_name].load(model_path)
        prediction = model.predict(dataset.features)
        analyzer = ResidualAnalyzer()
        prediction_stack = analyzer.restore_stack(dataset, prediction.y_pred)
        observed_stack = analyzer.restore_stack(dataset, dataset.targets)
        residuals = analyzer.analyze(dataset, prediction.y_pred)
        output_root = self._path(self.config["outputs"]["root"])
        prediction_dir = output_root / "predictions"
        self._write_predictions(
            prediction_dir, dataset, observed_stack, prediction_stack, prediction.uncertainty,
        )
        analyzer.write(residuals, dataset, output_root / "residuals")
        write_diagnostics(output_root / "residuals", dataset.targets, prediction.y_pred, dataset.dates, dataset.time_indices)
        detector = StatisticalAnomalyDetector(self.config["anomaly_detection"])
        anomalies = detector.detect(dataset, residuals.stack)
        detector.write(anomalies, dataset, output_root / "anomalies")
        result = {
            "prediction_count": int(prediction.y_pred.size), "residual_statistics": residuals.statistics,
            "anomaly_summary": anomalies.summary, "output_root": str(output_root),
        }
        LOGGER.info("MAP inference completed in %s", output_root)
        return result

    def _write_predictions(
        self,
        output_dir: Path,
        dataset: Any,
        observed_stack: np.ndarray,
        prediction_stack: np.ndarray,
        uncertainty: np.ndarray | None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = ResidualAnalyzer()
        for index, date in enumerate(dataset.dates):
            path = output_dir / f"prediction_{writer._safe_date(date)}.tif"
            writer._write_raster(path, prediction_stack[index], dataset, "baseline_prediction")
            observed_path = output_dir / f"observed_{writer._safe_date(date)}.tif"
            writer._write_raster(observed_path, observed_stack[index], dataset, "observed_deformation")
        if uncertainty is not None:
            uncertainty_stack = writer.restore_stack(dataset, uncertainty)
            for index, date in enumerate(dataset.dates):
                path = output_dir / f"uncertainty_{writer._safe_date(date)}.tif"
                writer._write_raster(path, uncertainty_stack[index], dataset, "prediction_uncertainty")

    def _feature_paths(self) -> list[Path]:
        data = self.config["data"]
        return [self._path(data["features_directory"]), self._path(data["temporal_features_directory"])]

    def _path(self, value: object) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.config_path.parent / path).resolve()

    def _name(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str):
            raise KeyError(f"Missing MAP configuration key: {key}")
        return value

    def _named_config(self, section: str, name: str) -> dict[str, Any]:
        value = self.config.get(section, {}).get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing configuration: {section}.{name}")
        return value


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible functional inference entry point."""
    return InferencePipeline(config).run()
