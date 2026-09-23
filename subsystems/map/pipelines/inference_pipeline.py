"""Scenario 1 inference, residual analysis and anomaly detection workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import DatasetBuilder, FeatureLoader
from subsystems.map.monitoring import ResidualAnalyzer
from subsystems.map.pipelines.monitoring_pipeline import (
    MONITORING_INPUT_FILENAME,
    MonitoringPipeline,
)
from subsystems.map.utils.artifacts import (
    write_diagnostics,
    write_json,
    write_latest_residual_map,
    write_mean_residual_map,
    write_observation_point_timeseries,
)
from subsystems.map.utils.experiment_paths import (
    experiment_model_directory,
    results_directory,
    static_file_path,
)
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
            self._feature_paths(),
            self._static_file(dataset_config['mask_file']),
            self._temporal_alignment_method(),
        ).load(
            list(dict.fromkeys([*feature_names, target_feature])),
            reference_feature=target_feature,
        )
        observation_points = self._observation_points(loaded.grid.crs)
        builder = DatasetBuilder()
        calibration_window = resolve_temporal_window(
            loaded.dates,
            dataset_config,
            'calibration',
            end_inclusive=False,
        )
        monitoring_window = resolve_temporal_window(
            loaded.dates,
            dataset_config,
            'monitoring',
        )
        fixed_support = self._fixed_support_mask(
            builder,
            loaded.features[target_feature],
            loaded.mask,
            calibration_window,
        )
        # This physical observation stack is deliberately created before any
        # LSTM sequence construction. It remains model-independent.
        observed_stack = np.where(
            fixed_support[np.newaxis, :, :],
            loaded.features[target_feature],
            np.nan,
        )
        dataset = builder.build(loaded, feature_names, target_feature)
        model_name = self._name('model')
        output_root = results_directory(self.config, self.config_path)
        model_path = (
            experiment_model_directory(
                output_root,
                self.config,
            )
            / 'model.pkl'
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                f'Model artifact not found for experiment {self.config["experiment"]["name"]!r}: '
                f'{model_path}. Run MAP learning with the same configuration first.',
            )
        model = MODEL_REGISTRY[model_name].load(model_path)
        specification = model.sequence_spec()
        if specification is not None:
            dataset = builder.build_sequences(dataset, *specification)
        prediction = model.predict(dataset.features)
        analyzer = ResidualAnalyzer()
        prediction_stack = analyzer.restore_stack(dataset, prediction.y_pred)
        residuals = analyzer.analyze(dataset, prediction.y_pred)
        prediction_dir = output_root / 'predictions'
        self._write_predictions(
            prediction_dir,
            dataset,
            observed_stack,
            prediction_stack,
            prediction.uncertainty,
        )
        analyzer.write(
            residuals,
            dataset,
            output_root / 'residuals',
            native_unit=self._native_plot_unit(),
            display_unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
        )
        write_latest_residual_map(
            output_dir=output_root / 'residuals',
            residual_stack=residuals.stack,
            dates=dataset.dates,
            mask=dataset.mask,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            points=observation_points,
            unit=self._plot_unit(),
            value_scale=self._plot_value_scale(),
            colormap=self._latest_residual_colormap(),
            percentile=self._latest_residual_percentile(),
            aggregation_window=self._latest_residual_window(),
            fallback_interval_days=self._fallback_interval_days(),
            cumulative_unit=self._cumulative_plot_unit(),
        )
        monitoring_residuals = residuals.stack[
            monitoring_window.start_index : monitoring_window.end_index
        ]
        monitoring_dates = dataset.dates[
            monitoring_window.start_index : monitoring_window.end_index
        ]
        write_latest_residual_map(
            output_dir=output_root / 'residuals',
            residual_stack=monitoring_residuals,
            dates=monitoring_dates,
            mask=dataset.mask,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            points=observation_points,
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
            points=observation_points,
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
            points=observation_points,
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
        monitoring_input = self._write_monitoring_input(
            output_root,
            dataset,
            observed_stack,
            prediction_stack,
            residuals.stack,
            None if prediction.uncertainty is None else analyzer.restore_stack(
                dataset, prediction.uncertainty
            ),
            fixed_support,
            calibration_window,
            monitoring_window,
        )
        run_after_inference = bool(
            self.config.get('monitoring', {}).get('run_after_inference', False)
        )
        monitoring_result = MonitoringPipeline(self.config).run() if run_after_inference else None
        result = {
            'prediction_count': int(prediction.y_pred.size),
            'residual_statistics_native': residuals.statistics,
            'native_deformation_rate_unit': self._native_plot_unit(),
            'display_deformation_rate_unit': self._plot_unit(),
            'value_scale_to_display_unit': self._plot_value_scale(),
            'monitoring_input_artifact': str(monitoring_input),
            'monitoring_ran': run_after_inference,
            'fixed_support_pixel_count': int(np.count_nonzero(fixed_support)),
            'fixed_support_fraction_of_tsf': float(
                np.count_nonzero(fixed_support) / np.count_nonzero(loaded.mask)
            ),
            'output_root': str(output_root),
            'dashboard_path': (
                None if monitoring_result is None else monitoring_result['dashboard_path']
            ),
        }
        write_json(output_root / 'inference_metadata.json', result)
        LOGGER.info('MAP inference completed in %s', output_root)
        return result

    def _write_monitoring_input(
        self,
        output_root: Path,
        dataset: Any,
        observed_stack: np.ndarray,
        prediction_stack: np.ndarray,
        residual_stack: np.ndarray,
        uncertainty_stack: np.ndarray | None,
        fixed_support: np.ndarray,
        calibration_window: Any,
        monitoring_window: Any,
    ) -> Path:
        """Persist the model-independent input contract for monitoring runs."""
        output_dir = output_root / 'inference'
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / MONITORING_INPUT_FILENAME
        np.savez_compressed(
            path,
            observed_stack=observed_stack,
            prediction_stack=prediction_stack,
            residual_stack=residual_stack,
            uncertainty_stack=(
                np.empty((0, 0, 0), dtype=np.float64)
                if uncertainty_stack is None
                else uncertainty_stack
            ),
            has_uncertainty=np.array(uncertainty_stack is not None),
            mask=dataset.mask,
            fixed_support_mask=fixed_support,
            dates=np.asarray(dataset.dates),
            calibration_window=np.array(
                [calibration_window.start_index, calibration_window.end_index], dtype=np.int64
            ),
            monitoring_window=np.array(
                [monitoring_window.start_index, monitoring_window.end_index], dtype=np.int64
            ),
            transform=np.asarray(tuple(dataset.grid.transform)[:6], dtype=np.float64),
            crs=np.asarray(str(dataset.grid.crs)),
            nodata=np.asarray(
                np.nan if dataset.grid.nodata is None else dataset.grid.nodata,
                dtype=np.float64,
            ),
        )
        return path

    def _fixed_support_mask(
        self,
        builder: DatasetBuilder,
        target_stack: np.ndarray,
        tsf_mask: np.ndarray,
        calibration_window: Any,
    ) -> np.ndarray:
        """Return configured model-independent support for physical monitoring."""
        dashboard = self._dashboard_config()
        spatial_support = dashboard.get('spatial_support', {})
        if not isinstance(spatial_support, dict):
            raise ValueError('monitoring.dashboard.spatial_support must be a mapping.')
        population = str(spatial_support.get('population', 'fixed_calibration_valid'))
        if population == 'fixed_calibration_valid':
            return builder.fixed_valid_mask(
                target_stack,
                tsf_mask,
                calibration_window.start_index,
                calibration_window.end_index,
                float(spatial_support.get('minimum_observation_coverage', 0.95)),
            )
        if population == 'tsf_mask':
            return tsf_mask.copy()
        raise ValueError(
            'monitoring.dashboard.spatial_support.population must be '
            '"fixed_calibration_valid" or "tsf_mask".',
        )

    def _spatial_coherence_config(self) -> dict[str, Any]:
        """Return the optional spatial coherence settings for anomaly gating."""
        dashboard = self._dashboard_config()
        value = dashboard.get('spatial_coherence', {})
        if not isinstance(value, dict):
            raise ValueError('monitoring.dashboard.spatial_coherence must be a mapping.')
        return value

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
                path,
                prediction_stack[index],
                dataset,
                'baseline_prediction_rate',
                unit=self._native_plot_unit(),
            )
            observed_path = output_dir / f'observed_{writer._safe_date(date)}.tif'
            writer._write_raster(
                observed_path,
                observed_stack[index],
                dataset,
                'observed_deformation_rate',
                unit=self._native_plot_unit(),
            )
        if uncertainty is not None:
            uncertainty_stack = writer.restore_stack(dataset, uncertainty)
            for index, date in enumerate(dataset.dates):
                path = output_dir / f'uncertainty_{writer._safe_date(date)}.tif'
                writer._write_raster(
                    path,
                    uncertainty_stack[index],
                    dataset,
                    'prediction_uncertainty_rate',
                    unit=self._native_plot_unit(),
                )

    def _feature_paths(self) -> list[Path]:
        """Return conventional DAG feature directories for this experiment."""
        root = results_directory(self.config, self.config_path)
        return [
            root / 'features',
            root / 'temporal_features',
            root / 'meteo_features',
        ]

    def _temporal_alignment_method(self) -> str:
        """Return configured causal temporal alignment for external features."""
        return str(self.config['data'].get('temporal_alignment_method', 'exact'))

    def _static_file(self, filename: object) -> Path:
        """Resolve a configured file name beneath the experiment static folder."""
        return static_file_path(self.config, self.config_path, filename)

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

    def _native_plot_unit(self) -> str:
        """Return the native rate unit stored in MAP raster artifacts."""
        return str(
            self.config.get('plotting', {}).get(
                'native_deformation_rate_unit',
                self._plot_unit(),
            )
        )

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

    def _observation_points(self, grid_crs: object) -> dict[str, dict[str, object]]:
        """Load configured diagnostic points in the monitored raster CRS.

        A GeoPackage/vector source is preferred, so points remain correct when a
        scenario is regenerated in a different projected coordinate system.
        The legacy inline mapping remains supported for small, self-contained
        configurations.
        """
        plotting = self.config.get('plotting', {})
        if not isinstance(plotting, dict):
            raise ValueError('plotting must be a mapping.')
        source_value = plotting.get('observation_points_file')
        if source_value is None:
            points = plotting.get('observation_points', {})
            if not isinstance(points, dict):
                raise ValueError('plotting.observation_points must be a mapping.')
            if not all(isinstance(value, dict) for value in points.values()):
                raise ValueError('Each configured observation point must be a mapping.')
            return points

        source_path = self._static_file(source_value)
        if not source_path.is_file():
            LOGGER.warning(
                'Observation-point file does not exist; skipping point diagnostics: %s',
                source_path,
            )
            return {}
        try:
            import geopandas as geopandas

            layer = plotting.get('observation_points_layer')
            point_frame = geopandas.read_file(
                source_path,
                **({} if layer is None else {'layer': str(layer)}),
            )
        except (ImportError, OSError, ValueError) as exc:
            LOGGER.warning(
                'Could not load observation points from %s; skipping point diagnostics: %s',
                source_path,
                exc,
            )
            return {}
        if point_frame.empty:
            LOGGER.warning('Observation-point file contains no features: %s', source_path)
            return {}
        if point_frame.crs is None:
            LOGGER.warning(
                'Observation-point file has no CRS; skipping point diagnostics: %s',
                source_path,
            )
            return {}
        try:
            point_frame = point_frame.to_crs(grid_crs)
        except (TypeError, ValueError) as exc:
            LOGGER.warning(
                'Could not reproject observation points to the raster CRS; '
                'skipping point diagnostics: %s',
                exc,
            )
            return {}
        name_field = str(plotting.get('observation_points_name_field', 'label'))
        if name_field not in point_frame.columns:
            LOGGER.warning(
                'Observation-point field %r is absent from %s; skipping point diagnostics.',
                name_field,
                source_path,
            )
            return {}
        points: dict[str, dict[str, object]] = {}
        for index, feature in point_frame.iterrows():
            name = str(feature[name_field]).strip()
            geometry = feature.geometry
            if not name or geometry is None or geometry.is_empty or geometry.geom_type != 'Point':
                LOGGER.warning('Skipping invalid observation-point feature %s.', index)
                continue
            if name in points:
                LOGGER.warning('Skipping duplicate observation-point name %r.', name)
                continue
            points[name] = {
                'coordinates': [float(geometry.x), float(geometry.y)],
            }
        if not points:
            LOGGER.warning('No valid point features were loaded from %s.', source_path)
        return points

    def _observation_window_size(self) -> int:
        """Return the configured odd pixel-window size for point diagnostics."""
        return int(self.config.get('plotting', {}).get('observation_window_size', 3))

    def _dashboard_config(self) -> dict[str, Any]:
        """Return the configured dashboard monitoring settings."""
        monitoring = self.config.get('monitoring', {})
        dashboard = (
            monitoring.get('dashboard') if isinstance(monitoring, dict) else None
        )
        if not isinstance(dashboard, dict):
            raise ValueError('monitoring.dashboard must be a mapping.')
        return dashboard

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

    def _persistent_fraction_display_max(self) -> float:
        """Return the fixed linear maximum for persistent-anomaly fractions."""
        return float(
            self.config.get('plotting', {}).get(
                'persistent_fraction_display_max',
                1.0,
            )
        )


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible functional inference entry point."""
    return InferencePipeline(config).run()
