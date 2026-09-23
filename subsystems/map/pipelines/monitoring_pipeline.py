"""Monitoring products derived from persisted MAP inference outputs."""

from __future__ import annotations

import logging
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
from subsystems.map.utils.artifacts import write_json, write_persistent_residual_map
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
        anomalies = detector.detect(
            dataset,
            residual_stack,
            persistence_start_time_index=calibration[1],
            persistence_end_time_index=monitoring[1],
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
            persistence_start_time_index=int(anomalies.summary['persistence_start_time_index']),
            persistence_end_time_index=int(anomalies.summary['persistence_end_time_index']),
            persistence_fraction_display_max=self._persistent_fraction_display_max(),
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

        dashboard_path: Path | None = None
        if bool(dashboard_config.get('enabled', True)):
            uncertainty = inputs['uncertainty_stack'] if inputs['has_uncertainty'] else None
            temporal = residual_monitor.analyze(
                observed_stack=inputs['observed_stack'],
                prediction_stack=inputs['prediction_stack'],
                dates=dataset.dates,
                calibration_window=calibration,
                monitoring_window=monitoring,
                uncertainty_stack=uncertainty,
                fixed_support_mask=inputs['fixed_support_mask'],
                coherent_regions=coherence.regions,
            )
            dashboard_path = self.output_root / 'monitoring' / str(
                dashboard_config['filename']
            )
            write_slope_stability_dashboard(
                output_path=dashboard_path,
                dates=dataset.dates,
                monitoring=temporal,
                observed_stack=inputs['observed_stack'],
                coherent_acceleration_cusum=coherent_cusum,
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
            inputs, calibration, monitoring, coherence.regions
        )

        result = {
            'input_artifact': str(self._input_path()),
            'anomaly_summary': anomalies.summary,
            'spatial_coherence_summary': coherence.summary,
            'dashboard_path': None if dashboard_path is None else str(dashboard_path),
            'maximum_coherence_acceleration_cusum_path': str(coherent_cusum_path),
            'animation_path': None if animation_path is None else str(animation_path),
            'animation_dashboard_data_path': (
                None if animation_path is None else str(animation_path.with_suffix('.json'))
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
    ) -> Path | None:
        """Optionally export a causal animation from persisted inference stacks."""
        config = self.config.get('monitoring', {}).get('animation', {})
        if not isinstance(config, dict):
            raise ValueError('monitoring.animation must be a mapping.')
        if not bool(config.get('enabled', False)):
            return None
        from subsystems.map.scripts.simulate_monitoring_animation import (
            configured_animation_output,
            export_dashboard_json,
            export_csv,
            run_precomputed_simulation,
            save_visualization,
        )

        dates = inputs['dates']
        calibration_window = TemporalWindow(
            calibration[0], calibration[1], dates[calibration[0]], dates[calibration[1] - 1]
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

    def _load_inputs(self) -> dict[str, Any]:
        """Load and validate the model-independent inference hand-off artifact."""
        path = self._input_path()
        if not path.is_file():
            raise FileNotFoundError(
                f'Monitoring input artifact not found: {path}. Run MAP inference first.'
            )
        with np.load(path, allow_pickle=False) as archive:
            required = {
                'observed_stack', 'prediction_stack', 'residual_stack',
                'uncertainty_stack', 'has_uncertainty', 'mask', 'fixed_support_mask',
                'dates', 'calibration_window', 'monitoring_window', 'transform',
                'crs', 'nodata',
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f'Monitoring input artifact is incomplete: {sorted(missing)}')
            inputs = {name: archive[name] for name in required}
        for name in ('observed_stack', 'prediction_stack', 'residual_stack'):
            if inputs[name].ndim != 3:
                raise ValueError(f'Monitoring input {name} must be a 3D stack.')
        if inputs['prediction_stack'].shape != inputs['observed_stack'].shape:
            raise ValueError('Monitoring observation and prediction stack shapes differ.')
        inputs['has_uncertainty'] = bool(inputs['has_uncertainty'].item())
        inputs['dates'] = tuple(str(value) for value in inputs['dates'])
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
            raise ValueError('monitoring.dashboard.spatial_coherence must be a mapping.')
        return value

    def _plot_unit(self) -> str:
        return str(self.config.get('plotting', {}).get('deformation_unit', ''))

    def _native_plot_unit(self) -> str:
        return str(self.config.get('plotting', {}).get('native_deformation_rate_unit', self._plot_unit()))

    def _plot_value_scale(self) -> float:
        return float(self.config.get('plotting', {}).get('value_scale', 1.0))

    def _persistent_residual_colormap(self) -> str:
        return str(self.config.get('plotting', {}).get('persistent_residual_colormap', 'magma'))

    def _persistent_fraction_display_max(self) -> float:
        return float(self.config.get('plotting', {}).get('persistent_fraction_display_max', 1.0))


def run_monitoring(config: dict[str, Any]) -> dict[str, Any]:
    """Functional entry point for the independent MAP monitoring workflow."""
    return MonitoringPipeline(config).run()
