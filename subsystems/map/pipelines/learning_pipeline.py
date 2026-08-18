"""Scenario 1 baseline-learning workflow."""

from __future__ import annotations

import logging
from pathlib import Path
import random
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictiveModel
from subsystems.map.core.registry import MODEL_REGISTRY, VARIABLE_REGISTRY
from subsystems.map.dataset import Dataset, DatasetBuilder, FeatureLoader
from subsystems.map.plugins.selection.stable_pixel_selector import StablePixelSelector
from subsystems.map.utils.artifacts import (
    regression_metrics,
    write_diagnostics,
    write_json,
    write_learning_curve,
)
from subsystems.map.utils.experiment_paths import experiment_model_directory
from subsystems.map.utils.explainability import write_tree_explainability
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
        if target_feature in feature_names:
            raise ValueError(
                'dataset.features must not contain dataset.target_feature; this '
                f'would leak the target into learning: {target_feature}.',
            )
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
        model = MODEL_REGISTRY[model_name](self._named_config('models', model_name))
        stable_dataset = self._sequence_dataset(builder, stable_dataset, model)
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
        model.set_random_seed(self._random_seed())
        model.set_validation_data(
            datasets.validation.features,
            datasets.validation.targets,
        )
        model.train(datasets.train.features, datasets.train.targets)
        validation = model.predict(datasets.validation.features).y_pred
        test = model.predict(datasets.test.features).y_pred
        output_root = self._path(self.config['outputs']['root'])
        models_dir = experiment_model_directory(output_root, self.config)
        model_path = models_dir / 'model.pkl'
        model.save(model_path)
        metrics = {
            'training': regression_metrics(
                datasets.train.targets, model.predict(datasets.train.features).y_pred
            ),
            'validation': regression_metrics(datasets.validation.targets, validation),
            'test': regression_metrics(datasets.test.targets, test),
        }
        write_json(models_dir / 'metrics.json', metrics)
        write_learning_curve(
            models_dir,
            list(getattr(model, 'training_history', [])),
            list(getattr(model, 'validation_history', [])),
        )
        write_diagnostics(
            models_dir,
            datasets.validation.targets,
            validation,
            datasets.validation.dates,
            datasets.validation.time_indices,
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
        )
        explainability = write_tree_explainability(
            models_dir,
            model,
            datasets.validation,
            self.config.get('explainability', {}),
            self._random_seed(),
            self._plot_unit(),
            self._plot_value_scale(),
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
            'model_artifact_directory': str(models_dir),
            'metrics': metrics,
            'explainability': explainability,
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
        seed = self._random_seed()
        random.seed(seed)
        np.random.seed(seed)

    def _random_seed(self) -> int:
        """Return the configured reproducibility seed."""
        return int(
            self.config.get('training', {}).get(
                'random_seed', self.config.get('experiment', {}).get('seed', 42)
            )
        )

    @staticmethod
    def _sequence_dataset(
        builder: DatasetBuilder,
        dataset: Dataset,
        model: PredictiveModel,
    ) -> Dataset:
        """Convert tabular samples only when a model declares sequence input."""
        specification = model.sequence_spec()
        if specification is None:
            return dataset
        look_back, horizon = specification
        return builder.build_sequences(dataset, look_back, horizon)

    def _plot_unit(self) -> str:
        """Return the configured physical unit used by diagnostic axes."""
        return str(self.config.get('plotting', {}).get('deformation_unit', ''))

    def _plot_value_scale(self) -> float:
        """Return the configured conversion from native values to plot units."""
        return float(self.config.get('plotting', {}).get('value_scale', 1.0))


def run_learning(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible functional learning entry point."""
    return LearningPipeline(config).run()
