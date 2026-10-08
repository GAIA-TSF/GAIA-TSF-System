from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.interfaces import Dataset
from dataset.feature_loader import load_feature_raster
from dataset.splitter import TemporalSplitter

logger = logging.getLogger("map.dataset_builder")


class DatasetBuilder:
    """Build a supervised dataset from engineered feature rasters."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = None

    def build(self) -> Dataset:
        """Load selected feature rasters and construct train/val/test arrays."""
        feature_config = self._feature_config()
        split_config = getattr(getattr(self.config, "dataset", None), "split", None)
        source_dir = Path(getattr(feature_config, "source_dir", "results/features"))
        selected_features = list(
            getattr(feature_config, "selected", None) or [getattr(feature_config, "target_feature", "velocity")]
        )
        target_feature = getattr(feature_config, "target_feature", selected_features[0])
        look_back = int(getattr(feature_config, "look_back", 1))

        loaded_arrays: list[np.ndarray] = []
        for feature_name in selected_features:
            feature_path = self._resolve_feature_path(source_dir, feature_name)
            loaded_arrays.append(load_feature_raster(feature_path))

        stacked = np.stack([self._prepare_array(values) for values in loaded_arrays], axis=0)
        windowed, sample_times, sample_pixels = self._build_windowed_features(stacked, look_back)
        target_values = self._prepare_array(load_feature_raster(self._resolve_feature_path(source_dir, target_feature)))
        target_windowed = self._build_targets(target_values, look_back)

        if windowed.shape[0] != target_windowed.shape[0]:
            target_windowed = target_windowed[: windowed.shape[0]]
            sample_times = sample_times[: windowed.shape[0]]
            sample_pixels = sample_pixels[: windowed.shape[0]]

        splitter = TemporalSplitter(
            train_ratio=float(getattr(split_config, "train_ratio", 0.7)),
            val_ratio=float(getattr(split_config, "val_ratio", 0.15)),
            test_ratio=float(getattr(split_config, "test_ratio", 0.15)),
        )
        n_temporal_samples = max(0, target_values.shape[0] - look_back + 1)
        splits = splitter.split(n_temporal_samples)
        train_rows = np.isin(sample_times, splits.train)
        val_rows = np.isin(sample_times, splits.val)
        test_rows = np.isin(sample_times, splits.test)

        return Dataset(
            X_train=windowed[train_rows],
            y_train=target_windowed[train_rows],
            X_val=windowed[val_rows],
            y_val=target_windowed[val_rows],
            X_test=windowed[test_rows],
            y_test=target_windowed[test_rows],
            feature_names=list(selected_features),
            metadata={
                "look_back": look_back,
                "target_feature": target_feature,
                "feature_pipeline": self._active_feature_pipeline(),
                "source_dir": str(source_dir),
            },
            train_time_indices=sample_times[train_rows],
            val_time_indices=sample_times[val_rows],
            test_time_indices=sample_times[test_rows],
            train_pixel_indices=sample_pixels[train_rows],
            val_pixel_indices=sample_pixels[val_rows],
            test_pixel_indices=sample_pixels[test_rows],
            stable_selection_values=target_values,
            raster_shape=self._raster_shape(loaded_arrays[0]),
        )

    def _build_windowed_features(self, stacked: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create pixel-time feature rows from a stacked temporal array."""
        feature_count = stacked.shape[0]
        time_steps = stacked.shape[1]
        pixel_count = stacked.shape[2]
        rows: list[np.ndarray] = []
        time_indices: list[int] = []
        pixel_indices: list[int] = []

        for sample_index in range(look_back - 1, time_steps):
            split_time_index = sample_index - look_back + 1
            for pixel_index in range(pixel_count):
                feature_parts = [
                    stacked[feature_index, sample_index - look_back + 1 : sample_index + 1, pixel_index]
                    for feature_index in range(feature_count)
                ]
                rows.append(np.concatenate(feature_parts))
                time_indices.append(split_time_index)
                pixel_indices.append(pixel_index)

        if not rows:
            return (
                np.empty((0, feature_count * look_back)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
            )
        return np.vstack(rows), np.asarray(time_indices, dtype=int), np.asarray(pixel_indices, dtype=int)

    def _build_targets(self, target_values: np.ndarray, look_back: int) -> np.ndarray:
        """Create scalar target rows aligned with the windowed feature rows."""
        targets = []
        for sample_index in range(look_back - 1, target_values.shape[0]):
            targets.extend(target_values[sample_index, :].tolist())
        return np.asarray(targets, dtype=float)

    def _prepare_array(self, values: np.ndarray) -> np.ndarray:
        """Flatten a feature array to a 2D representation shaped as (time, pixels)."""
        values = np.asarray(values)
        if values.ndim == 1:
            return values[:, None]
        if values.ndim == 2:
            return values
        return values.reshape(values.shape[0], -1)

    def _resolve_feature_path(self, source_dir: Path, feature_name: str) -> Path:
        """Resolve a feature path from a feature name and the configured source directory."""
        candidates = [
            source_dir / feature_name,
            source_dir / f"{feature_name}.npy",
            source_dir / f"{feature_name}.npz",
            source_dir / f"{feature_name}.tif",
            source_dir / f"{feature_name}.tiff",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not locate feature '{feature_name}' in {source_dir}")

    def _active_feature_pipeline(self) -> str:
        """Return the configured feature pipeline name."""
        return str(
            getattr(self.config, "active_feature_pipeline", None)
            or getattr(self.config, "feature_pipeline", None)
            or "temporal"
        )

    def _feature_config(self) -> Any:
        """Return the config section for the active feature pipeline."""
        features_config = getattr(self.config, "features", None)
        if features_config is None:
            raise ValueError("Missing 'features' configuration.")

        pipeline_name = self._active_feature_pipeline()
        pipeline_config = getattr(features_config, pipeline_name, None)
        if pipeline_config is not None:
            return pipeline_config
        if getattr(features_config, "selected", None) is not None:
            return features_config
        raise ValueError(f"Missing feature pipeline configuration for '{pipeline_name}'.")

    def _raster_shape(self, values: np.ndarray) -> tuple[int, ...] | None:
        """Return the spatial shape of a temporal raster, if one is available."""
        values = np.asarray(values)
        if values.ndim <= 2:
            return None
        return tuple(values.shape[1:])
