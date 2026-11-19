"""MAP training and chronological validation of one forecasting model."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import rasterio

from subsystems.map.core.interfaces import PredictiveModel
from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import (
    FeatureLoader,
    ForecastSamples,
    LoadedFeatures,
    SampleSplits,
    build_forecast_samples,
    split_spatial,
    split_temporal,
)
from subsystems.map.plugins.selection.stable_pixel_selector import StablePixelSelector
from subsystems.map.utils.artifacts import (
    write_error_by_lead,
    write_learning_curve,
    write_observed_vs_predicted,
)
from subsystems.map.utils.config_loader import MapContext
from subsystems.map.utils.explainability import write_tree_explainability
from subsystems.map.utils.forecasts import (
    apply_interval,
    forecast_columns,
    grid_metadata,
    interval_offsets,
    lead_metrics,
    write_json,
    write_table,
)
from subsystems.map.utils.temporal_windows import resolve_temporal_window


LOGGER = logging.getLogger(__name__)
SELECTION_METHODS = ('temporal_std', 'configured_mask', 'configured_mask_and_temporal_std')


def load_kv_features(ctx: MapContext, extra: tuple[str, ...] = ()) -> LoadedFeatures:
    """Load predictors and target on the target's acquisition axis."""
    names = list(dict.fromkeys([*ctx.feature_names, ctx.target_feature, *extra]))
    return FeatureLoader(
        ctx.feature_dirs,
        ctx.static_file(ctx.dataset['mask_file']),
        ctx.temporal_alignment_method,
    ).load(names, reference_feature=ctx.target_feature)


def resolve_windows(ctx: MapContext, dates: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Calibration (exclusive end) and monitoring index windows."""
    calibration = resolve_temporal_window(dates, ctx.dataset, 'calibration', end_inclusive=False)
    monitoring = resolve_temporal_window(dates, ctx.dataset, 'monitoring')
    return {
        'calibration': (calibration.start_index, calibration.end_index),
        'monitoring': (monitoring.start_index, monitoring.end_index),
    }


def predict_values(model: PredictiveModel, samples: ForecastSamples, chunk: int = 100000) -> np.ndarray:
    """Absolute forecasts: persistence anchor plus the predicted change."""
    changes = [
        model.predict(samples.features[start : start + chunk])
        for start in range(0, max(len(samples.features), 1), chunk)
    ]
    return samples.anchors[:, None] + np.concatenate(changes, axis=0)


class LearningPipeline:
    """Train one model on reference (normal) data and validate it per lead."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.ctx = MapContext.from_config(config)

    def run(self) -> dict[str, Any]:
        import subsystems.map.plugins.models  # noqa: F401

        ctx = self.ctx
        loaded, windows, reference_mask, selection_metadata = self.load_reference()
        start, end = windows['calibration']
        model = MODEL_REGISTRY[ctx.model_name](ctx.model_config, ctx.horizon)
        samples, splits, method = self.build_splits(
            loaded, reference_mask, windows, model.look_back()
        )

        random.seed(ctx.seed)
        np.random.seed(ctx.seed)
        model.set_random_seed(ctx.seed)
        model.set_validation_data(splits.validation.features, splits.validation.changes)
        model.train(splits.train.features, splits.train.changes)

        predicted = {name: predict_values(model, subset) for name, subset in splits.items()}
        uncertainty = interval_offsets(
            splits.validation.targets, predicted['validation'], ctx.interval_level
        )
        model_dir = ctx.model_dir
        model_path = model_dir / 'model.pkl'
        model.save(model_path)
        model_version = (
            f'{ctx.experiment_name}@{hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]}'
        )
        identity = {**ctx.identity(), 'model_version': model_version}

        metrics: dict[str, list[dict[str, Any]]] = {}
        reference_tables = []
        for name, subset in splits.items():
            lower, upper = apply_interval(predicted[name], uncertainty)
            metrics[name] = lead_metrics(subset, predicted[name], lower, upper)
            columns = forecast_columns(
                subset, predicted[name], lower, upper, windows, ctx.acquisition_interval_days
            )
            columns['sample_role'][:] = name
            columns['observed_value'] = subset.targets.ravel()
            columns['residual'] = columns['observed_value'] - columns['predicted_value']
            observed = np.isfinite(columns['observed_value'])
            reference_tables.append({key: value[observed] for key, value in columns.items()})
        reference = {
            key: np.concatenate([table[key] for table in reference_tables])
            for key in reference_tables[0]
        }
        write_table(
            model_dir / 'reference_forecasts',
            reference,
            {
                **identity,
                **grid_metadata(samples),
                'product': 'map_reference_forecast_errors',
                'description': (
                    'Forecasts on the reference (normal) population inside the '
                    'calibration window. validation and test rows are held-out '
                    'normal forecast errors; train rows were seen in fitting.'
                ),
                'interval_level': ctx.interval_level,
            },
        )
        self._write_metrics_csv(model_dir / 'validation_metrics.csv', identity, metrics)

        if ctx.plotting:
            label = f'{ctx.experiment_name}: {method} validation'
            write_learning_curve(
                model_dir,
                list(getattr(model, 'training_history', [])),
                list(getattr(model, 'validation_history', [])),
            )
            write_observed_vs_predicted(
                model_dir, splits.validation.targets, predicted['validation'],
                ctx.dataset['target_units'], label,
            )
            write_error_by_lead(model_dir, metrics['validation'], ctx.dataset['target_units'], label)
        explainability = self._explainability(model, splits.validation)

        metadata = {
            **identity,
            'feature_names': ctx.feature_names,
            'model_config': ctx.model_config,
            'look_back': model.look_back(),
            'model_path': str(model_path),
            'reference_selection': selection_metadata,
            'split_method': method,
            'calibration_window': {
                'start_date': loaded.dates[start],
                'end_date': loaded.dates[end - 1],
            },
            'split_date_ranges': {
                name: {
                    'start_date': loaded.dates[int(subset.origin_indices.min())],
                    'end_date': loaded.dates[int(subset.origin_indices.max())],
                    'origin_count': int(np.unique(subset.origin_indices).size),
                    'pixel_count': int(np.unique(subset.pixel_indices).size),
                    'sample_count': int(subset.origin_indices.size),
                }
                for name, subset in splits.items()
            },
            'uncertainty': uncertainty,
            'metrics': metrics,
            'explainability': explainability,
        }
        write_json(model_dir / 'experiment.json', metadata)
        LOGGER.info('MAP learning completed: %s', model_path)
        return metadata

    def load_reference(self) -> tuple[LoadedFeatures, dict[str, tuple[int, int]], np.ndarray, dict[str, Any]]:
        """Load features, resolve windows, and select the reference population."""
        selection = self._selection()
        extra = (selection['std_feature'],) if selection['uses_std'] else ()
        loaded = load_kv_features(self.ctx, extra)
        windows = resolve_windows(self.ctx, loaded.dates)
        start, end = windows['calibration']
        mask, metadata = self._calibration_mask(selection, loaded, start, end)
        return loaded, windows, mask, metadata

    def build_splits(
        self,
        loaded: LoadedFeatures,
        reference_mask: np.ndarray,
        windows: dict[str, tuple[int, int]],
        look_back: int | None,
    ) -> tuple[ForecastSamples, SampleSplits, str]:
        """Reference forecast samples and their configured train/validation/test split."""
        ctx = self.ctx
        start, end = windows['calibration']
        samples = build_forecast_samples(
            replace(loaded, mask=reference_mask),
            ctx.feature_names,
            ctx.target_feature,
            ctx.horizon,
            look_back,
            origin_range=(start, end),
        )
        split = ctx.dataset.get('split', {})
        method = str(split.get('method', 'temporal'))
        ratios = tuple(float(split[key]) for key in ('train_ratio', 'validation_ratio', 'test_ratio'))
        if method == 'temporal':
            splits = split_temporal(samples, start, end, *ratios)
        elif method == 'spatial':
            splits = split_spatial(
                samples, start, end, *ratios, int(split.get('random_seed', ctx.seed))
            )
        else:
            raise ValueError('split.method must be one of: spatial, temporal.')
        return samples, splits, method

    def _selection(self) -> dict[str, Any]:
        """Validated reference-population selection settings."""
        selection = self.ctx.dataset.get('calibration_spatial_selection', {})
        if not isinstance(selection, dict):
            raise TypeError('calibration_spatial_selection must be a mapping.')
        method = str(selection.get('method', 'configured_mask'))
        if method not in SELECTION_METHODS:
            raise ValueError(
                'calibration_spatial_selection.method must be one of: '
                + ', '.join(SELECTION_METHODS)
            )
        uses_std = method != 'configured_mask'
        if uses_std and selection.get('std_threshold') is None:
            raise KeyError(f'{method} requires calibration_spatial_selection.std_threshold.')
        return {
            'method': method,
            'uses_std': uses_std,
            'std_feature': str(selection.get('std_feature', self.ctx.target_feature)),
            'std_threshold': None if not uses_std else float(selection['std_threshold']),
            'mask_file': selection.get('mask_file', self.ctx.dataset.get('stable_file')),
        }

    def _calibration_mask(
        self, selection: dict[str, Any], loaded: Any, start: int, end: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select the spatial reference population used only for training."""
        method = selection['method']
        configured_path = None
        domain = loaded.mask
        if method != 'temporal_std':
            filename = selection['mask_file']
            if not isinstance(filename, str) or not filename.strip():
                raise KeyError(
                    'Configured-mask calibration requires stable_file or '
                    'calibration_spatial_selection.mask_file.'
                )
            configured_path = self.ctx.static_file(filename)
            domain = self._load_calibration_mask(configured_path, loaded)
        mask = domain
        if selection['uses_std']:
            mask = StablePixelSelector(selection['std_threshold']).select(
                loaded.features[selection['std_feature']][start:end], domain
            )
        return mask, {
            'method': method,
            'mask_file': None if configured_path is None else str(configured_path),
            'configured_pixel_count': (
                None if configured_path is None else int(np.count_nonzero(domain))
            ),
            'selected_pixel_count': int(np.count_nonzero(mask)),
            'std_feature': selection['std_feature'] if selection['uses_std'] else None,
            'std_threshold': selection['std_threshold'],
            'selection_start_date': loaded.dates[start],
            'selection_end_date': loaded.dates[end - 1],
        }

    @staticmethod
    def _load_calibration_mask(path: Path, loaded: Any) -> np.ndarray:
        """Load a binary stable-area mask on the already validated feature grid."""
        if not path.exists():
            raise FileNotFoundError(f'Calibration stable mask does not exist: {path}')
        with rasterio.open(path) as source:
            values = source.read(1, masked=True).filled(0)
            grid = (source.height, source.width, source.transform, source.crs)
        expected = (loaded.grid.height, loaded.grid.width, loaded.grid.transform, loaded.grid.crs)
        if grid != expected:
            raise ValueError(f'Calibration stable mask grid does not match feature grid: {path}')
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError('Calibration stable mask must be binary (0/1).')
        mask = values == 1
        if not np.any(mask):
            raise ValueError('Calibration stable mask contains no selected pixels.')
        return mask

    def _explainability(self, model: PredictiveModel, validation: ForecastSamples) -> dict[str, Any]:
        """Tree explainability per lead for TRF, when enabled."""
        config = self.ctx.config.get('explainability', {})
        if not config.get('enabled', False):
            return {'status': 'disabled'}
        if not hasattr(model, 'models'):
            return {'status': 'not_supported', 'reason': 'Only TRF is a tree ensemble.'}
        result = {}
        changes = validation.changes
        for lead, estimator in enumerate(model.models, start=1):
            observed = np.isfinite(changes[:, lead - 1])
            result[f'lead_{lead}'] = write_tree_explainability(
                self.ctx.model_dir / 'explainability' / f'lead_{lead}',
                estimator,
                validation.features[observed],
                changes[observed, lead - 1],
                validation.origin_indices[observed],
                list(validation.feature_names),
                config,
                self.ctx.seed,
                str(self.ctx.dataset['target_units']),
                1.0,
            )
        return result

    @staticmethod
    def _write_metrics_csv(
        path: Path, identity: dict[str, Any], metrics: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Errors by KV, target, model, split, and lead in one flat table."""
        keys = ('kv', 'target_variable', 'target_units', 'model', 'experiment', 'model_version')
        rows = [
            {**{key: identity[key] for key in keys}, 'split': split, **row}
            for split, values in metrics.items()
            for row in values
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_learning(config: dict[str, Any]) -> dict[str, Any]:
    """Functional learning entry point."""
    return LearningPipeline(config).run()
