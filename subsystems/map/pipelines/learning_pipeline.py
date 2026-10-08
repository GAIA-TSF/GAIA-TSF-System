"""Scenario 1 baseline-learning workflow."""

from __future__ import annotations

import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import rasterio

from subsystems.map.core.interfaces import PredictiveModel
from subsystems.map.core.registry import MODEL_REGISTRY, VARIABLE_REGISTRY
from subsystems.map.dataset import DatasetBuilder, FeatureLoader
from subsystems.map.plugins.selection.stable_pixel_selector import StablePixelSelector
from subsystems.map.utils.artifacts import (
    regression_metrics,
    write_diagnostics,
    write_json,
    write_learning_curve,
)
from subsystems.map.utils.experiment_paths import (
    experiment_model_directory,
    results_directory,
    static_file_path,
)
from subsystems.map.utils.explainability import write_tree_explainability
from subsystems.map.utils.temporal_windows import resolve_temporal_window


LOGGER = logging.getLogger(__name__)


class LearningPipeline:
    """Train a registered predictive model from stable pixel samples."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config_path = Path(str(config['_config_path']))

    def run(self) -> dict[str, Any]:
        """Select stable pixels, train, validate,
        persist and register the experiment."""
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
                'would leak the target into learning.',
            )
        loaded = FeatureLoader(
            self._feature_paths(),
            self._static_file(dataset_config['mask_file']),
            self._temporal_alignment_method(),
        ).load(
            list(dict.fromkeys([*feature_names, target_feature])),
            reference_feature=target_feature,
        )
        spatial_calibration_window = resolve_temporal_window(
            loaded.dates, dataset_config, 'calibration', end_inclusive=False
        )
        stable_mask, calibration_selection = self._calibration_mask(
            dataset_config, loaded, target_feature,
            spatial_calibration_window.start_index,
            spatial_calibration_window.end_index,
        )
        builder = DatasetBuilder()
        stable_dataset = builder.build(
            loaded, feature_names, target_feature, stable_mask
        )
        model = MODEL_REGISTRY[model_name](self._named_config('models', model_name))
        stable_dataset = self._sequence_dataset(builder, stable_dataset, model)
        split = dataset_config['split']
        split_method = str(split.get('method', 'temporal'))
        calibration_window = resolve_temporal_window(
            stable_dataset.dates,
            dataset_config,
            'calibration',
            end_inclusive=False,
        )
        split_arguments = (
            stable_dataset,
            calibration_window.start_index,
            calibration_window.end_index,
            float(split['train_ratio']),
            float(split['validation_ratio']),
            float(split['test_ratio']),
        )
        if split_method == 'temporal':
            datasets = builder.split_temporal_window(*split_arguments)
        elif split_method == 'spatial':
            datasets = builder.split_spatial_window(
                *split_arguments,
                random_seed=int(split.get('random_seed', self._random_seed())),
            )
        else:
            raise ValueError(
                'dataset.split.method must be one of: spatial, temporal.'
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
        output_root = results_directory(self.config, self.config_path)
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
            dataset_label=f'{split_method.capitalize()} validation subset',
        )
        explainability_config = self.config.get('explainability', {})
        if not isinstance(explainability_config, dict):
            raise ValueError('explainability must be a mapping.')
        explainability = write_tree_explainability(
            models_dir,
            model,
            datasets.validation,
            explainability_config,
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
            'calibration_spatial_selection': calibration_selection,
            'split_method': split_method,
            'calibration_window': {
                'start_date': calibration_window.start_date,
                'end_date': calibration_window.end_date,
            },
            'sample_counts': {
                'train': int(datasets.train.targets.size),
                'validation': int(datasets.validation.targets.size),
                'test': int(datasets.test.targets.size),
            },
            'split_pixel_counts': {
                'train': int(np.unique(datasets.train.pixel_indices).size),
                'validation': int(np.unique(datasets.validation.pixel_indices).size),
                'test': int(np.unique(datasets.test.pixel_indices).size),
            },
            'split_date_ranges': {
                'train': self._dataset_date_range(datasets.train),
                'validation': self._dataset_date_range(datasets.validation),
                'test': self._dataset_date_range(datasets.test),
            },
            'model_path': str(model_path),
            'explainability': explainability,
            'metrics': metrics,
        }
        write_json(models_dir / 'experiment.json', metadata)
        LOGGER.info('MAP learning completed: %s', model_path)
        return metadata

    def _feature_paths(self) -> list[Path]:
        """Return conventional DAG feature directories for this experiment."""
        root = results_directory(self.config, self.config_path)
        return [
            root / 'features',
            root / 'temporal_features',
            root / 'meteo_features',
        ]

    def _calibration_mask(
        self,
        dataset_config: dict[str, Any],
        loaded: Any,
        target_feature: str,
        calibration_start: int,
        calibration_end: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select the spatial population used exclusively for model calibration."""
        selection = dataset_config.get('calibration_spatial_selection', {})
        if not isinstance(selection, dict):
            raise TypeError('dataset.calibration_spatial_selection must be a mapping.')
        method = str(selection.get('method', 'temporal_std'))
        supported = {
            'temporal_std', 'configured_mask',
            'configured_mask_and_temporal_std',
        }
        if method not in supported:
            raise ValueError(
                'calibration_spatial_selection.method must be one of: '
                + ', '.join(sorted(supported))
            )
        threshold = float(self.config['baseline_model']['stable_pixel_std_threshold'])
        selector = StablePixelSelector(threshold)
        configured_path = None
        configured_count = None
        if method == 'temporal_std':
            mask = selector.select(
                loaded.features[target_feature][calibration_start:calibration_end],
                loaded.mask,
            )
        else:
            stable_file = selection.get('mask_file', dataset_config.get('stable_file'))
            if not isinstance(stable_file, str) or not stable_file.strip():
                raise KeyError(
                    'Configured-mask calibration requires dataset.stable_file or '
                    'calibration_spatial_selection.mask_file.'
                )
            configured_path = self._static_file(stable_file)
            configured = self._load_calibration_mask(configured_path, loaded)
            outside = configured & ~loaded.mask
            if np.any(outside):
                raise ValueError(
                    f'Calibration stable mask contains {np.count_nonzero(outside)} '
                    'pixels outside dataset.mask_file.'
                )
            configured_count = int(np.count_nonzero(configured))
            if method == 'configured_mask':
                mask = configured & loaded.mask
            else:
                mask = selector.select(
                    loaded.features[target_feature][calibration_start:calibration_end],
                    configured & loaded.mask,
                )
        return mask, {
            'method': method,
            'mask_file': None if configured_path is None else str(configured_path),
            'configured_pixel_count': configured_count,
            'selected_pixel_count': int(np.count_nonzero(mask)),
            'temporal_std_threshold': (
                threshold if method != 'configured_mask' else None
            ),
            'monitoring_domain': 'dataset.mask_file',
            'selection_start_date': loaded.dates[calibration_start],
            'selection_end_date': loaded.dates[calibration_end - 1],
        }

    @staticmethod
    def _load_calibration_mask(path: Path, loaded: Any) -> np.ndarray:
        """Load a binary stable-area mask on the already validated feature grid."""
        if not path.exists():
            raise FileNotFoundError(f'Calibration stable mask does not exist: {path}')
        with rasterio.open(path) as source:
            values = source.read(1, masked=True).filled(0)
            grid = (source.height, source.width, source.transform, source.crs)
        expected = (
            loaded.grid.height, loaded.grid.width,
            loaded.grid.transform, loaded.grid.crs,
        )
        if grid != expected:
            raise ValueError(
                f'Calibration stable mask grid does not match feature grid: {path}'
            )
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError('Calibration stable mask must be binary (0/1).')
        mask = values == 1
        if not np.any(mask):
            raise ValueError('Calibration stable mask contains no selected pixels.')
        return mask

    def _temporal_alignment_method(self) -> str:
        """Return the configured causal feature-date alignment method."""
        return str(self.config['data'].get('temporal_alignment_method', 'exact'))

    def _static_file(self, filename: object) -> Path:
        """Resolve a configured file name beneath the experiment static folder."""
        return static_file_path(self.config, self.config_path, filename)

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
        dataset: Any,
        model: PredictiveModel,
    ) -> Any:
        """Build causal sequences only for a model that declares them."""
        specification = model.sequence_spec()
        if specification is None:
            return dataset
        return builder.build_sequences(dataset, *specification)

    def _plot_unit(self) -> str:
        """Return the configured display unit for diagnostic plots."""
        return str(self.config.get('plotting', {}).get('deformation_unit', ''))

    def _plot_value_scale(self) -> float:
        """Return the native-to-display scale for diagnostic plots."""
        return float(self.config.get('plotting', {}).get('value_scale', 1.0))

    @staticmethod
    def _dataset_date_range(dataset: Any) -> dict[str, object]:
        """Describe acquisition dates that actually contribute samples."""
        indices = np.unique(dataset.time_indices)
        return {
            'start_date': dataset.dates[int(indices[0])],
            'end_date': dataset.dates[int(indices[-1])],
            'acquisition_count': int(indices.size),
        }


def run_learning(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible functional learning entry point."""
    return LearningPipeline(config).run()
