"""Scenario 1 inference, residual analysis and anomaly detection workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import DatasetBuilder, FeatureLoader
from subsystems.map.monitoring import ResidualAnalyzer, StatisticalAnomalyDetector
from subsystems.map.utils.artifacts import (
    write_diagnostics,
    write_latest_residual_map,
    write_mean_residual_map,
    write_observation_point_timeseries,
    write_persistent_residual_map,
)
from subsystems.map.utils.experiment_paths import experiment_model_directory
from subsystems.map.utils.temporal_windows import resolve_temporal_window


LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    """Predict every valid TSF pixel, then derive residual and anomaly products."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config_path = Path(str(config['_config_path']))

    def run(self) -> dict[str, Any]:
        """Run inference across the configured TSF mask and persist all products."""
        import subsystems.map.plugins.models  # noqa: F401

        dataset_config = self._named_config('datasets', self._name('dataset'))
        feature_names = [str(value) for value in dataset_config['features']]
        target_feature = str(dataset_config['target_feature'])
        loaded = FeatureLoader(
            self._feature_paths(), self._path(dataset_config['mask_path'])
        ).load(
            list(dict.fromkeys([*feature_names, target_feature])),
        )
        dataset = DatasetBuilder().build(loaded, feature_names, target_feature)
        calibration_window = resolve_temporal_window(
            dataset.dates,
            dataset_config,
            'calibration',
        )
        monitoring_window = resolve_temporal_window(
            dataset.dates,
            dataset_config,
            'monitoring',
        )
        model_name = self._name('model')
        output_root = self._path(self.config['outputs']['root'])
        model_path = experiment_model_directory(
            output_root,
            self.config,
        ) / 'model.pkl'
        if not model_path.is_file():
            raise FileNotFoundError(
                f'Model artifact not found for experiment {self.config["experiment"]["name"]!r}: '
                f'{model_path}. Run MAP learning with the same configuration first.',
            )
        model = MODEL_REGISTRY[model_name].load(model_path)
        prediction = model.predict(dataset.features)
        analyzer = ResidualAnalyzer()
        prediction_stack = analyzer.restore_stack(dataset, prediction.y_pred)
        observed_stack = analyzer.restore_stack(dataset, dataset.targets)
        residuals = analyzer.analyze(dataset, prediction.y_pred)
        prediction_dir = output_root / 'predictions'
        self._write_predictions(
            prediction_dir,
            dataset,
            observed_stack,
            prediction_stack,
            prediction.uncertainty,
        )
        analyzer.write(residuals, dataset, output_root / 'residuals')
        write_latest_residual_map(
            output_dir=output_root / 'residuals',
            residual_stack=residuals.stack,
            dates=dataset.dates,
            mask=dataset.mask,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            points=self._observation_points(),
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
            colormap=self._latest_residual_colormap(),
            percentile=self._latest_residual_percentile(),
            aggregation_window=self._latest_residual_window(),
            fallback_interval_days=self._fallback_interval_days(),
            cumulative_unit=self._cumulative_plot_unit(),
        )
        monitoring_residuals = residuals.stack[
            monitoring_window.start_index:monitoring_window.end_index
        ]
        monitoring_dates = dataset.dates[
            monitoring_window.start_index:monitoring_window.end_index
        ]
        write_latest_residual_map(
            output_dir=output_root / 'residuals',
            residual_stack=monitoring_residuals,
            dates=monitoring_dates,
            mask=dataset.mask,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            points=self._observation_points(),
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
            colormap=self._latest_residual_colormap(),
            percentile=self._latest_residual_percentile(),
            aggregation_window=monitoring_residuals.shape[0],
            fallback_interval_days=self._fallback_interval_days(),
            cumulative_unit=self._cumulative_plot_unit(),
            output_filename='residual_monitoring.png',
            title_override=(
                'Cumulative TSF residual during monitoring period '
                f'({monitoring_dates[0]} to {monitoring_dates[-1]})'
            ),
        )
        write_mean_residual_map(
            output_dir=output_root / 'residuals',
            residual_stack=residuals.stack,
            dates=dataset.dates,
            mask=dataset.mask,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            points=self._observation_points(),
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
            colormap=self._mean_residual_colormap(),
            percentile=self._mean_residual_percentile(),
        )
        write_diagnostics(
            output_root / 'residuals',
            dataset.targets,
            prediction.y_pred,
            dataset.dates,
            dataset.time_indices,
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
            uncertainty=prediction.uncertainty,
            calibration_window=(
                calibration_window.start_index,
                calibration_window.end_index,
            ),
            monitoring_window=(
                monitoring_window.start_index,
                monitoring_window.end_index,
            ),
            cumulative_unit=self._cumulative_plot_unit(),
            fallback_interval_days=self._fallback_interval_days(),
            pixel_indices=dataset.pixel_indices,
            cumulative_observation_max_points=self._cumulative_observation_max_points(),
        )
        write_observation_point_timeseries(
            output_dir=output_root / 'residuals',
            observed=dataset.targets,
            dates=dataset.dates,
            time_indices=dataset.time_indices,
            pixel_indices=dataset.pixel_indices,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            points=self._observation_points(),
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
            cumulative_unit=self._cumulative_plot_unit(),
            fallback_interval_days=self._fallback_interval_days(),
            window_size=self._observation_window_size(),
            calibration_window=(
                calibration_window.start_index,
                calibration_window.end_index,
            ),
            monitoring_window=(
                monitoring_window.start_index,
                monitoring_window.end_index,
            ),
        )
        detector = StatisticalAnomalyDetector(self.config['anomaly_detection'])
        anomalies = detector.detect(
            dataset,
            residuals.stack,
            persistence_start_time_index=monitoring_window.start_index,
            persistence_end_time_index=monitoring_window.end_index,
        )
        detector.write(anomalies, dataset, output_root / 'anomalies')
        write_persistent_residual_map(
            output_dir=output_root / 'residuals',
            persistent_anomalies=anomalies.binary_stack,
            mask=dataset.mask,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            points=self._observation_points(),
            colormap=self._persistent_residual_colormap(),
            persistence_start_time_index=int(
                anomalies.summary['persistence_start_time_index'],
            ),
            persistence_end_time_index=int(
                anomalies.summary['persistence_end_time_index'],
            ),
            fraction_gamma=self._persistent_fraction_gamma(),
        )
        result = {
            'prediction_count': int(prediction.y_pred.size),
            'residual_statistics': residuals.statistics,
            'anomaly_summary': anomalies.summary,
            'output_root': str(output_root),
        }
        LOGGER.info('MAP inference completed in %s', output_root)
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
            path = output_dir / f'prediction_{writer._safe_date(date)}.tif'
            writer._write_raster(
                path, prediction_stack[index], dataset, 'baseline_prediction'
            )
            observed_path = output_dir / f'observed_{writer._safe_date(date)}.tif'
            writer._write_raster(
                observed_path, observed_stack[index], dataset, 'observed_deformation'
            )
        if uncertainty is not None:
            uncertainty_stack = writer.restore_stack(dataset, uncertainty)
            for index, date in enumerate(dataset.dates):
                path = output_dir / f'uncertainty_{writer._safe_date(date)}.tif'
                writer._write_raster(
                    path, uncertainty_stack[index], dataset, 'prediction_uncertainty'
                )

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

    def _name(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str):
            raise KeyError(f'Missing MAP configuration key: {key}')
        return value

    def _named_config(self, section: str, name: str) -> dict[str, Any]:
        value = self.config.get(section, {}).get(name)
        if not isinstance(value, dict):
            raise KeyError(f'Missing configuration: {section}.{name}')
        return value

    def _plot_unit(self) -> str:
        """Return the configured physical unit used by diagnostic axes."""
        return str(self.config.get('plotting', {}).get('deformation_unit', ''))

    def _plot_value_scale(self) -> float:
        """Return the configured conversion from native values to plot units."""
        return float(self.config.get('plotting', {}).get('value_scale', 1.0))

    def _cumulative_plot_unit(self) -> str | None:
        """Return the configured cumulative-displacement unit, if enabled."""
        value = self.config.get('plotting', {}).get('cumulative_displacement_unit')
        return None if value is None else str(value)

    def _fallback_interval_days(self) -> float:
        """Return the temporal interval used for non-date acquisition labels."""
        return float(self.config.get('plotting', {}).get('fallback_interval_days', 1.0))

    def _cumulative_observation_max_points(self) -> int | None:
        """Return the optional rendering cap for cumulative observation samples."""
        value = self.config.get('plotting', {}).get(
            'cumulative_observation_max_points',
        )
        return None if value is None else int(value)

    def _observation_points(self) -> dict[str, dict[str, object]]:
        """Return named point coordinates configured for observation diagnostics."""
        points = self.config.get('plotting', {}).get('observation_points', {})
        if not isinstance(points, dict):
            raise ValueError('plotting.observation_points must be a mapping.')
        if not all(isinstance(value, dict) for value in points.values()):
            raise ValueError('Each configured observation point must be a mapping.')
        return points

    def _observation_window_size(self) -> int:
        """Return the configured odd pixel-window size for point diagnostics."""
        return int(self.config.get('plotting', {}).get('observation_window_size', 3))

    def _latest_residual_colormap(self) -> str:
        """Return the configured diverging colour map for the latest residual map."""
        return str(
            self.config.get('plotting', {}).get(
                'latest_residual_colormap',
                'RdBu_r',
            )
        )

    def _latest_residual_percentile(self) -> float:
        """Return the absolute-residual percentile used for colour scaling."""
        return float(
            self.config.get('plotting', {}).get(
                'latest_residual_percentile',
                98.0,
            )
        )

    def _latest_residual_window(self) -> int:
        """Return the number of latest residual acquisitions to integrate."""
        return int(
            self.config.get('plotting', {}).get('latest_residual_window', 1),
        )

    def _mean_residual_colormap(self) -> str:
        """Return the configured diverging colour map for the mean residual map."""
        return str(
            self.config.get('plotting', {}).get(
                'mean_residual_colormap',
                'RdBu_r',
            )
        )

    def _mean_residual_percentile(self) -> float:
        """Return the configured colour-scale percentile for mean residuals."""
        return float(
            self.config.get('plotting', {}).get(
                'mean_residual_percentile',
                98.0,
            )
        )

    def _persistent_residual_colormap(self) -> str:
        """Return the configured sequential colour map for persistent anomalies."""
        return str(
            self.config.get('plotting', {}).get(
                'persistent_residual_colormap',
                'magma',
            )
        )

    def _persistent_fraction_gamma(self) -> float:
        """Return the visual emphasis applied to low persistence fractions."""
        return float(
            self.config.get('plotting', {}).get(
                'persistent_fraction_gamma',
                1.0,
            )
        )



def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible functional inference entry point."""
    return InferencePipeline(config).run()
