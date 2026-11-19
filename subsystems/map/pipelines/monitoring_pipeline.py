"""Monitoring products derived from persisted MAP inference outputs."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.dataset.dataset_builder import Dataset
from subsystems.map.dataset.feature_loader import RasterGrid
from subsystems.map.monitoring import (
    ResidualAnalyzer,
    SpatialCoherenceDetector,
    StatisticalAnomalyDetector,
    TemporalResidualMonitor,
)
from subsystems.map.monitoring.dashboard import write_slope_stability_dashboard
from subsystems.map.monitoring.level1 import FORMULA, frozen_residual_calibration
from subsystems.map.monitoring.level2 import (
    Level2Parameters,
    compute_level2,
    frozen_level2_calibration,
    level2_metadata,
    pixel_level2,
)
from subsystems.map.monitoring.quality import (
    QualityParameters,
    acquisition_quality,
    calibration_date_counts,
    run_quality_flags,
)
from subsystems.map.monitoring.series_export import (
    build_records,
    phases,
    write_series,
)
from subsystems.map.monitoring.level3 import (
    Level3Parameters,
    compute_level3,
    frozen_level3_calibration,
    level3_metadata,
)
from subsystems.map.utils.artifacts import write_json, write_persistent_residual_map
from subsystems.map.utils.contract import CONTRACT_VERSION, SCHEMA_VERSION, UNKNOWN
from subsystems.map.utils.forecast_timing import lead_time_summary
from subsystems.map.utils.experiment_paths import results_directory
from subsystems.map.utils.temporal_windows import TemporalWindow


LOGGER = logging.getLogger(__name__)
MONITORING_INPUT_FILENAME = 'monitoring_input.npz'


class MonitoringPipeline:
    """Create anomaly, coherence, dashboard, and animation-ready products.

    The pipeline intentionally consumes the inference hand-off artifact rather
    than feature rasters or a trained model.  Operational threshold changes can
    therefore be evaluated reproducibly without recalculating predictions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Store MAP configuration and resolve its experiment-local results path."""
        self.config = config
        self.config_path = Path(str(config['_config_path']))
        self.output_root = results_directory(config, self.config_path)

    def run(self) -> dict[str, Any]:
        """Run monitoring from the persisted inference hand-off artifact."""
        inputs = self._load_inputs()
        dataset = self._dataset(inputs)
        calibration = tuple(int(value) for value in inputs['calibration_window'])
        monitoring = tuple(int(value) for value in inputs['monitoring_window'])
        residual_stack = inputs['residual_stack']
        detector = StatisticalAnomalyDetector(self.config['anomaly_detection'])
        reused = self._existing_calibrations()
        calibration_path = Path(str(self.config['anomaly_detection'].get(
            'calibration_file', 'monitoring/l1_calibration.json'
        )))
        if not calibration_path.is_absolute():
            calibration_path = self.output_root / calibration_path
        l1_calibration = frozen_residual_calibration(
            calibration_path, residual_stack, dataset.dates, calibration, monitoring[0],
            # Use the static TSF mask. Observation-derived fixed_support_mask
            # may have been selected over a longer, retrospective window.
            dataset.mask,
            unit=self._native_plot_unit(), target_feature=dataset.target_feature,
        )
        level1 = {
            'formula': FORMULA,
            'threshold': detector.zscore_threshold,
            'direction': detector.direction,
            'comparison': 'strict',
            'calibration': l1_calibration,
        }
        level2_parameters = Level2Parameters.from_config(self.config)
        l2_calibration_path = self._results_path(
            self.config.get('level2', {}).get(
                'calibration_file', 'monitoring/l2_calibration.json'
            )
        )
        l2_calibration = frozen_level2_calibration(
            l2_calibration_path, residual_stack, dataset.dates, calibration,
            monitoring[0], dataset.mask,
            unit=self._native_plot_unit(), target_feature=dataset.target_feature,
            parameters=level2_parameters,
        )
        anomalies = detector.detect(
            dataset,
            residual_stack,
            persistence_start_time_index=monitoring[0],
            persistence_end_time_index=monitoring[1],
            calibration=l1_calibration,
        )
        detector.write(
            anomalies,
            dataset,
            self.output_root / 'anomalies',
            residual_rate_unit=self._native_plot_unit(),
        )
        coherence = SpatialCoherenceDetector(self._spatial_coherence_config()).detect(
            anomalies.binary_stack,
            dataset.mask,
        )
        self._write_coherent_anomalies(coherence.binary_stack, dataset)
        write_json(
            self.output_root / 'anomalies' / 'spatial_coherence_summary.json',
            coherence.summary,
        )
        write_persistent_residual_map(
            output_dir=self.output_root / 'residuals',
            persistent_anomalies=anomalies.binary_stack,
            mask=dataset.mask,
            grid_transform=dataset.grid.transform,
            grid_width=dataset.grid.width,
            grid_height=dataset.grid.height,
            colormap=self._persistent_residual_colormap(),
            persistence_start_time_index=int(
                anomalies.summary['persistence_start_time_index']
            ),
            persistence_end_time_index=int(
                anomalies.summary['persistence_end_time_index']
            ),
            persistence_fraction_display_max=self._persistent_fraction_display_max(),
        )

        level2 = compute_level2(
            residual_stack, dataset.mask, calibration, monitoring,
            l2_calibration, level2_parameters, coherence.regions,
        )
        level2_summary = level2_metadata(level2, dataset.dates)
        pixel_l2_paths = self._write_pixel_level2(
            residual_stack, l1_calibration['sigma'], dataset, monitoring,
            level2_parameters,
        )

        dashboard_config = self._dashboard_config()
        residual_monitor = TemporalResidualMonitor(dashboard_config)
        coherent_cusum = residual_monitor.spatial_coherent_acceleration_cusum(
            inputs['observed_stack'],
            dataset.dates,
            calibration,
            monitoring,
            coherence.binary_stack,
        )
        coherent_cusum_path = (
            self.output_root / 'monitoring' / 'maximum_coherence_acceleration_cusum.tif'
        )
        coherent_cusum_path.parent.mkdir(parents=True, exist_ok=True)
        ResidualAnalyzer()._write_raster(
            coherent_cusum_path,
            coherent_cusum,
            dataset,
            'maximum_coherence_qualified_acceleration_cusum',
            unit='dimensionless',
        )

        # L3 needs the temporal series even when the dashboard is disabled.
        temporal = residual_monitor.analyze(
            observed_stack=inputs['observed_stack'],
            prediction_stack=inputs['prediction_stack'],
            dates=dataset.dates,
            calibration_window=calibration,
            monitoring_window=monitoring,
            uncertainty_stack=(
                inputs['uncertainty_stack'] if inputs['has_uncertainty'] else None
            ),
            fixed_support_mask=inputs['fixed_support_mask'],
            coherent_regions=coherence.regions,
        )
        l3_calibration_path = self._results_path(
            self.config.get('level3', {}).get(
                'calibration_file', 'monitoring/l3_calibration.json'
            )
        )
        l3_calibration = frozen_level3_calibration(
            l3_calibration_path, temporal.regime_calibration, dataset.dates,
            unit=self._native_plot_unit(), target_feature=dataset.target_feature,
        )
        level3 = compute_level3(
            temporal.regime_change_score,
            temporal.regional_regime_change_score,
            level2,
            monitoring,
            Level3Parameters.from_config(self.config),
            coherence.regions,
            calibration=l3_calibration,
            calibration_scores=temporal.regime_calibration_score,
        )
        level3_summary = level3_metadata(level3, dataset.dates)

        quality_parameters = QualityParameters.from_config(self.config)
        quality_states, coverage = acquisition_quality(
            inputs['observed_stack'],
            inputs['prediction_stack'],
            dataset.mask,
            monitoring,
            quality_parameters,
        )
        run_mode = inputs.get('run_mode', UNKNOWN)
        quality_flags = run_quality_flags(
            run_mode=run_mode,
            l2_bias_warning=bool(level2_summary['bias_warning']),
            calibration_dates=calibration_date_counts(
                l1_calibration, l2_calibration, l3_calibration
            ),
            calibrations_reused=reused,
            parameters=quality_parameters,
        )
        timing = self._timing_columns(inputs, dataset.dates)
        series_paths = write_series(
            self.output_root / 'monitoring' / 'monitoring_series',
            build_records(
                {
                    **timing,
                    'phase': phases(len(dataset.dates), calibration, monitoring),
                    'quality_state': quality_states,
                    'observed_tsf_coverage': coverage,
                    **self._temporal_columns(temporal),
                    'l1_anomalous_pixels': self._pixel_counts(
                        anomalies.instantaneous_stack, len(dataset.dates)
                    ),
                    'l1_persistent_anomalous_pixels': self._pixel_counts(
                        anomalies.binary_stack, len(dataset.dates)
                    ),
                    'l1_coherent_anomalous_pixels': self._pixel_counts(
                        coherence.binary_stack, len(dataset.dates)
                    ),
                    'l2_z': level2.z,
                    'l2_instability_cusum': level2.instability_cusum,
                    'l2_overforecast_cusum': level2.overforecast_cusum,
                    'l2_alarm': level2.alarm,
                    'coherent_region_available': level2.regional_available,
                    'coherent_region_l2_instability_cusum': (
                        level2.regional_instability_cusum
                    ),
                    'coherent_region_l2_overforecast_cusum': (
                        level2.regional_overforecast_cusum
                    ),
                    'coherent_region_l2_alarm': level2.regional_alarm,
                    'regime_change_score': level3.score,
                    'coherent_region_regime_change_score': level3.regional_score,
                    'l3_level': level3.level,
                    **{
                        f'l3_{name}': flags
                        for name, flags in level3.criteria.items()
                    },
                },
                range(len(dataset.dates)),
            ),
            rate_unit=self._native_plot_unit(),
            metadata={
                'run_mode': run_mode,
                'target_feature': dataset.target_feature,
                'calibration_window': list(calibration),
                'monitoring_window': list(monitoring),
            },
        )

        dashboard_path: Path | None = None
        if bool(dashboard_config.get('enabled', True)):
            dashboard_path = (
                self.output_root / 'monitoring' / str(dashboard_config['filename'])
            )
            write_slope_stability_dashboard(
                output_path=dashboard_path,
                dates=dataset.dates,
                monitoring=temporal,
                observed_stack=inputs['observed_stack'],
                coherent_acceleration_cusum=coherent_cusum,
                level2=level2,
                level3=level3,
                pixel_level2_maximum=pixel_l2_paths['values'],
                mask=dataset.mask,
                grid_transform=dataset.grid.transform,
                grid_width=dataset.grid.width,
                grid_height=dataset.grid.height,
                unit=self._plot_unit(),
                value_scale=self._plot_value_scale(),
                calibration_window=calibration,
                monitoring_window=monitoring,
            )

        animation_path = self._write_animation(
            inputs, calibration, monitoring, coherence.regions,
            level1=level1, level2=level2, level3=level3,
        )

        result = {
            'schema_version': SCHEMA_VERSION,
            'contract_version': CONTRACT_VERSION,
            'run_mode': run_mode,
            'run_mode_reason': inputs.get('run_mode_reason', ''),
            'lead_time': lead_time_summary(
                [
                    {
                        'lead_time_days': days,
                        'lead_time_acquisitions': acquisitions,
                    }
                    for days, acquisitions in zip(
                        timing['lead_time_days'], timing['lead_time_acquisitions']
                    )
                ]
            ),
            'quality': {
                'minimum_coverage': quality_parameters.minimum_coverage,
                'minimum_calibration_samples': (
                    quality_parameters.minimum_calibration_samples
                ),
                'state_counts_monitoring': {
                    state: int(np.sum(quality_states[monitoring[0] : monitoring[1]] == state))
                    for state in ('ok', 'warm_up', 'missing_acquisition', 'insufficient_coverage')
                },
                'run_quality_flags': quality_flags,
            },
            'monitoring_series_path': str(series_paths[0]),
            'monitoring_series_csv_path': str(series_paths[1]),
            'forecast_timing_schema_version': inputs.get('forecast_timing_schema_version', 'legacy_unknown'),
            'target_feature': inputs.get('target_feature', ''),
            'input_artifact': str(self._input_path()),
            'anomaly_summary': anomalies.summary,
            'level1': level1,
            'l1_calibration_path': str(calibration_path),
            'level2': level2_summary,
            'l2_calibration_path': str(l2_calibration_path),
            'level3': level3_summary,
            'l3_calibration_path': str(l3_calibration_path),
            'pixel_level2_maximum_cusum_path': str(pixel_l2_paths['maximum']),
            'pixel_level2_first_alarm_index_path': str(pixel_l2_paths['first_alarm']),
            # Lookup for acquisition-index rasters such as the L2 first alarm.
            'acquisition_dates': list(dataset.dates),
            'spatial_coherence_summary': coherence.summary,
            'dashboard_path': None if dashboard_path is None else str(dashboard_path),
            'maximum_coherence_acceleration_cusum_path': str(coherent_cusum_path),
            'animation_path': None if animation_path is None else str(animation_path),
            'animation_dashboard_data_path': (
                None
                if animation_path is None
                else str(animation_path.with_suffix('.json'))
            ),
        }
        write_json(self.output_root / 'monitoring' / 'monitoring_metadata.json', result)
        LOGGER.info('MAP monitoring completed in %s', self.output_root / 'monitoring')
        return result

    def _write_animation(
        self,
        inputs: dict[str, Any],
        calibration: tuple[int, int],
        monitoring: tuple[int, int],
        coherent_regions: tuple[Any, ...],
        *,
        level1: dict[str, Any] | None = None,
        level2: Any = None,
        level3: Any = None,
    ) -> Path | None:
        """Optionally export a causal animation from persisted inference stacks."""
        config = self.config.get('monitoring', {}).get('animation', {})
        if not isinstance(config, dict):
            raise ValueError('monitoring.animation must be a mapping.')
        if not bool(config.get('enabled', False)):
            return None
        from subsystems.map.scripts.visualisation.simulate_monitoring_animation import (
            configured_animation_output,
            export_dashboard_json,
            export_csv,
            run_precomputed_simulation,
            save_visualization,
        )

        dates = inputs['dates']
        calibration_window = TemporalWindow(
            calibration[0],
            calibration[1],
            dates[calibration[0]],
            dates[calibration[1] - 1],
        )
        monitoring_window = TemporalWindow(
            monitoring[0], monitoring[1], dates[monitoring[0]], dates[monitoring[1] - 1]
        )
        result = run_precomputed_simulation(
            self.config,
            dates=dates,
            calibration=calibration_window,
            monitoring=monitoring_window,
            observed_stack=inputs['observed_stack'],
            prediction_stack=inputs['prediction_stack'],
            uncertainty_stack=(
                inputs['uncertainty_stack'] if inputs['has_uncertainty'] else None
            ),
            fixed_support_mask=inputs['fixed_support_mask'],
            coherent_regions=coherent_regions,
            forecast_origin_dates=inputs.get('forecast_origin_dates', ()),
            level1=level1,
            level2=level2,
            level3=level3,
        )
        output = configured_animation_output(self.config)
        saved = save_visualization(
            result,
            output,
            fps=int(config.get('fps', 4)),
            dpi=int(config.get('dpi', 150)),
            show=False,
        )
        export_csv(result, saved)
        dashboard_data_path = export_dashboard_json(result, saved)
        LOGGER.info('Wrote animation dashboard graph data: %s', dashboard_data_path)
        return saved

    def _existing_calibrations(self) -> list[str]:
        """Levels whose frozen calibration artifact predates this run."""
        files = {
            'l1': self.config['anomaly_detection'].get(
                'calibration_file', 'monitoring/l1_calibration.json'
            ),
            'l2': self.config.get('level2', {}).get(
                'calibration_file', 'monitoring/l2_calibration.json'
            ),
            'l3': self.config.get('level3', {}).get(
                'calibration_file', 'monitoring/l3_calibration.json'
            ),
        }
        return [level for level, name in files.items() if self._results_path(name).exists()]

    @staticmethod
    def _pixel_counts(stack: np.ndarray | None, size: int) -> np.ndarray:
        """Flagged pixels per acquisition; NaN when the product is absent."""
        if stack is None:
            return np.full(size, np.nan)
        return np.count_nonzero(stack, axis=(1, 2))

    @staticmethod
    def _timing_columns(
        inputs: dict[str, Any], dates: tuple[str, ...]
    ) -> dict[str, list[Any]]:
        """Forecast origin and lead time per target date (contract 3.2)."""
        origins = inputs.get('forecast_origin_dates') or ('',) * len(dates)
        index = {value: position for position, value in enumerate(dates)}
        days: list[int | None] = []
        acquisitions: list[int | None] = []
        for position, (origin, target) in enumerate(zip(origins, dates)):
            if not origin:
                days.append(None)
                acquisitions.append(None)
                continue
            days.append((date.fromisoformat(target) - date.fromisoformat(origin)).days)
            acquisitions.append(position - index[origin] if origin in index else None)
        return {
            'date': list(dates),
            'forecast_origin_date': [origin or None for origin in origins],
            'lead_time_days': days,
            'lead_time_acquisitions': acquisitions,
        }

    @staticmethod
    def _temporal_columns(temporal: Any) -> dict[str, Any]:
        """Aggregate rates and supporting acceleration evidence, native units."""
        spread = temporal.uncertainty_mean
        return {
            'observed_mean_los_velocity': temporal.observed_mean,
            'predicted_baseline_velocity': temporal.predicted_mean,
            'prediction_spread': (
                np.full(temporal.observed_mean.size, np.nan) if spread is None else spread
            ),
            'mean_residual_rate': temporal.residual_mean,
            'acceleration_cusum_input': temporal.acceleration,
            'acceleration_cusum_instability': temporal.acceleration_cusum_instability,
            'acceleration_cusum_recovery': temporal.acceleration_cusum_recovery,
            'dynamics': temporal.dynamics,
            'oscillation': temporal.oscillation,
            'coherent_region_acceleration_cusum_instability': (
                temporal.regional_acceleration_cusum_instability
            ),
            'coherent_region_acceleration_cusum_recovery': (
                temporal.regional_acceleration_cusum_recovery
            ),
            'coherent_region_dynamics': np.where(
                temporal.regional_cusum_available, temporal.regional_dynamics, ''
            ),
        }

    def _results_path(self, value: Any) -> Path:
        """Resolve a configured path relative to the experiment results."""
        path = Path(str(value))
        return path if path.is_absolute() else self.output_root / path

    def _write_pixel_level2(
        self,
        residual_stack: np.ndarray,
        sigma: float,
        dataset: Dataset,
        monitoring: tuple[int, int],
        parameters: Level2Parameters,
    ) -> dict[str, Any]:
        """Write per-pixel L2 rasters: maximum CUSUM and first alarm index."""
        maximum, first_alarm = pixel_level2(
            residual_stack, sigma, dataset.mask, monitoring, parameters
        )
        directory = self.output_root / 'monitoring'
        directory.mkdir(parents=True, exist_ok=True)
        maximum_path = directory / 'level2_maximum_instability_cusum.tif'
        first_alarm_path = directory / 'level2_first_alarm_index.tif'
        writer = ResidualAnalyzer()
        writer._write_raster(
            maximum_path, maximum, dataset,
            'maximum_monitoring_period_level2_instability_cusum',
            unit='dimensionless',
        )
        # Index into monitoring_metadata dates; no-data where no alarm occurred.
        writer._write_raster(
            first_alarm_path, first_alarm, dataset,
            'first_level2_alarm_acquisition_index',
            unit='acquisition_index',
        )
        return {
            'values': maximum,
            'maximum': maximum_path,
            'first_alarm': first_alarm_path,
        }

    def _load_inputs(self) -> dict[str, Any]:
        """Load and validate the model-independent inference hand-off artifact."""
        path = self._input_path()
        if not path.is_file():
            raise FileNotFoundError(
                f'Monitoring input artifact not found: {path}. Run MAP inference first.'
            )
        with np.load(path, allow_pickle=False) as archive:
            required = {
                'observed_stack',
                'prediction_stack',
                'residual_stack',
                'uncertainty_stack',
                'has_uncertainty',
                'mask',
                'fixed_support_mask',
                'dates',
                'calibration_window',
                'monitoring_window',
                'transform',
                'crs',
                'nodata',
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f'Monitoring input artifact is incomplete: {sorted(missing)}'
                )
            inputs = {name: archive[name] for name in required}
            for name in (
                'target_dates', 'forecast_origin_dates', 'source_label_dates',
                'target_feature', 'forecast_timing_schema_version',
                'run_mode', 'run_mode_reason',
            ):
                if name in archive.files:
                    inputs[name] = archive[name]
        for name in ('observed_stack', 'prediction_stack', 'residual_stack'):
            if inputs[name].ndim != 3:
                raise ValueError(f'Monitoring input {name} must be a 3D stack.')
        if inputs['prediction_stack'].shape != inputs['observed_stack'].shape:
            raise ValueError(
                'Monitoring observation and prediction stack shapes differ.'
            )
        inputs['has_uncertainty'] = bool(inputs['has_uncertainty'].item())
        inputs['dates'] = tuple(str(value) for value in inputs['dates'])
        for name in ('target_dates', 'forecast_origin_dates', 'source_label_dates'):
            if name in inputs:
                if inputs[name].shape != (len(inputs['dates']),):
                    raise ValueError(f'Monitoring input {name} must match dates.')
                inputs[name] = tuple(str(value) for value in inputs[name])
        if 'target_dates' in inputs and inputs['target_dates'] != inputs['dates']:
            raise ValueError('Monitoring date axis must match forecast target dates.')
        for name in (
            'target_feature', 'forecast_timing_schema_version',
            'run_mode', 'run_mode_reason',
        ):
            if name in inputs:
                inputs[name] = str(inputs[name].item())
        return inputs

    def _dataset(self, inputs: dict[str, Any]) -> Dataset:
        """Recreate the minimal spatial dataset required by GIS writers."""
        from affine import Affine

        transform = Affine(*[float(value) for value in inputs['transform']])
        grid = RasterGrid(
            crs=str(inputs['crs'].item()),
            transform=transform,
            height=int(inputs['mask'].shape[0]),
            width=int(inputs['mask'].shape[1]),
            nodata=float(inputs['nodata'].item()),
        )
        return Dataset(
            features=np.empty((0, 0), dtype=np.float64),
            targets=np.empty(0, dtype=np.float64),
            time_indices=np.empty(0, dtype=np.int64),
            pixel_indices=np.empty(0, dtype=np.int64),
            feature_names=(),
            dates=inputs['dates'],
            grid=grid,
            mask=inputs['mask'].astype(bool),
            source_dates=inputs.get('source_label_dates', ()),
            forecast_origin_dates=inputs.get('forecast_origin_dates', ()),
            target_feature=inputs.get('target_feature', ''),
        )

    def _write_coherent_anomalies(self, stack: np.ndarray, dataset: Dataset) -> None:
        """Write coherence-qualified anomaly rasters for GIS consumers."""
        analyzer = ResidualAnalyzer()
        output_dir = self.output_root / 'anomalies'
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, date in enumerate(dataset.dates):
            analyzer._write_raster(
                output_dir / f'anomaly_coherent_{analyzer._safe_date(date)}.tif',
                stack[index].astype(float),
                dataset,
                'spatially_coherent_anomaly_binary',
            )

    def _input_path(self) -> Path:
        """Return the inference artifact consumed by this pipeline."""
        return self.output_root / 'inference' / MONITORING_INPUT_FILENAME

    def _dashboard_config(self) -> dict[str, Any]:
        monitoring = self.config.get('monitoring', {})
        value = monitoring.get('dashboard') if isinstance(monitoring, dict) else None
        if not isinstance(value, dict):
            raise ValueError('monitoring.dashboard must be a mapping.')
        return value

    def _spatial_coherence_config(self) -> dict[str, Any]:
        value = self._dashboard_config().get('spatial_coherence', {})
        if not isinstance(value, dict):
            raise ValueError(
                'monitoring.dashboard.spatial_coherence must be a mapping.'
            )
        return value

    def _plot_unit(self) -> str:
        return str(self.config.get('plotting', {}).get('deformation_unit', ''))

    def _native_plot_unit(self) -> str:
        return str(
            self.config.get('plotting', {}).get(
                'native_deformation_rate_unit', self._plot_unit()
            )
        )

    def _plot_value_scale(self) -> float:
        return float(self.config.get('plotting', {}).get('value_scale', 1.0))

    def _persistent_residual_colormap(self) -> str:
        return str(
            self.config.get('plotting', {}).get('persistent_residual_colormap', 'magma')
        )

    def _persistent_fraction_display_max(self) -> float:
        return float(
            self.config.get('plotting', {}).get('persistent_fraction_display_max', 1.0)
        )


def run_monitoring(config: dict[str, Any]) -> dict[str, Any]:
    """Functional entry point for the independent MAP monitoring workflow."""
    return MonitoringPipeline(config).run()
