"""Scenario 1 baseline-learning workflow."""

from __future__ import annotations

import logging
from pathlib import Path
import random
from typing import Any

import numpy as np

from subsystems.map.core.registry import MODEL_REGISTRY, VARIABLE_REGISTRY
from subsystems.map.dataset import DatasetBuilder, FeatureLoader
from subsystems.map.plugins.selection.stable_pixel_selector import StablePixelSelector
from subsystems.map.utils.artifacts import (
    regression_metrics,
    write_diagnostics,
    write_json,
)
from subsystems.map.utils.temporal_windows import resolve_temporal_window


LOGGER = logging.getLogger(__name__)


class LearningPipeline:
    """Train a registered predictive model from stable pixel samples."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config_path = Path(str(config['_config_path']))

    def run(self) -> dict[str, Any]:
        """Select stable pixels, train, validate, persist and register the experiment."""
        import subsystems.map.plugins.models  # noqa: F401
        import subsystems.map.plugins.variables  # noqa: F401

        variable_name = self._required_name('variable')
        model_name = self._required_name('model')
        variable = VARIABLE_REGISTRY[variable_name]
        if model_name not in variable.allowed_models():
            raise ValueError(
                f"Model '{model_name}' is not allowed for variable '{variable_name}'."
            )
        dataset_config = self._named_config('datasets', self._required_name('dataset'))
        feature_names = [str(name) for name in dataset_config['features']]
        target_feature = str(dataset_config['target_feature'])
        loaded = FeatureLoader(
            self._feature_paths(), self._path(dataset_config['mask_path'])
        ).load(
            list(dict.fromkeys([*feature_names, target_feature])),
        )
        selector = StablePixelSelector(
            float(self.config['baseline_model']['stable_pixel_std_threshold'])
        )
        stable_mask = selector.select(loaded.features[target_feature], loaded.mask)
        builder = DatasetBuilder()
        stable_dataset = builder.build(
            loaded, feature_names, target_feature, stable_mask
        )
        split = dataset_config['split']
        if split.get('method') != 'temporal':
            raise ValueError('Scenario 1 supports only dataset.split.method: temporal.')
        calibration_window = resolve_temporal_window(
            stable_dataset.dates,
            dataset_config,
            'calibration',
        )
        datasets = builder.split_temporal_window(
            stable_dataset,
            calibration_window.start_index,
            calibration_window.end_index,
            float(split['train_ratio']),
            float(split['validation_ratio']),
            float(split['test_ratio']),
        )
        self._seed()
        model = MODEL_REGISTRY[model_name](self._named_config('models', model_name))
        model.train(datasets.train.features, datasets.train.targets)
        validation = model.predict(datasets.validation.features).y_pred
        test = model.predict(datasets.test.features).y_pred
        output_root = self._path(self.config['outputs']['root'])
        models_dir = output_root / 'models'
        model_path = models_dir / 'baseline_model.pkl'
        model.save(model_path)
        metrics = {
            'training': regression_metrics(
                datasets.train.targets, model.predict(datasets.train.features).y_pred
            ),
            'validation': regression_metrics(datasets.validation.targets, validation),
            'test': regression_metrics(datasets.test.targets, test),
        }
        write_json(models_dir / 'metrics.json', metrics)
        write_diagnostics(
            models_dir,
            datasets.validation.targets,
            validation,
            datasets.validation.dates,
            datasets.validation.time_indices,
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
        )
        metadata = {
            'experiment': self.config.get('experiment', {}),
            'variable': variable_name,
            'model': model_name,
            'feature_names': feature_names,
            'target_feature': target_feature,
            'stable_pixel_count': int(np.count_nonzero(stable_mask)),
            'calibration_window': {
                'start_date': calibration_window.start_date,
                'end_date': calibration_window.end_date,
            },
            'sample_counts': {
                'train': int(datasets.train.targets.size),
                'validation': int(datasets.validation.targets.size),
                'test': int(datasets.test.targets.size),
            },
            'model_path': str(model_path),
            'metrics': metrics,
        }
        write_json(models_dir / 'experiment.json', metadata)
        LOGGER.info('MAP learning completed: %s', model_path)
        return metadata

    def _feature_paths(self) -> list[Path]:
        data = self.config['data']
        return [
            self._path(data['features_directory']),
            self._path(data['temporal_features_directory']),
        ]

    def _path(self, value: object) -> Path:
        path = Path(str(value)).expanduser()
        return (
            path if path.is_absolute() else (self.config_path.parent / path).resolve()
        )

    def _required_name(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str):
            raise KeyError(f'Missing MAP configuration key: {key}')
        return value

    def _named_config(self, section: str, name: str) -> dict[str, Any]:
        value = self.config.get(section, {}).get(name)
        if not isinstance(value, dict):
            raise KeyError(f'Missing configuration: {section}.{name}')
        return value

    def _seed(self) -> None:
        seed = int(
            self.config.get('training', {}).get(
                'random_seed', self.config.get('experiment', {}).get('seed', 42)
            )
        )
        random.seed(seed)
        np.random.seed(seed)

    def _plot_unit(self) -> str:
        """Return the configured physical unit used by diagnostic axes."""
        return str(self.config.get('plotting', {}).get('deformation_unit', ''))

    def _plot_value_scale(self) -> float:
        """Return the configured conversion from native values to plot units."""
        return float(self.config.get('plotting', {}).get('value_scale', 1.0))


def run_learning(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible functional learning entry point."""
    return LearningPipeline(config).run()
