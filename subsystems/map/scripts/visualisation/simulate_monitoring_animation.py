"""Animate causal MAP calibration and acquisition-by-acquisition monitoring.

Usage:
python3 subsystems/map/scripts/visualisation/simulate_monitoring_animation.py \
    --config subsystems/map/config.yaml \
    --reuse-model \
    --fps 4 \
    --dpi 150

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import Dataset, DatasetBuilder, FeatureLoader
from subsystems.map.monitoring import ResidualAnalyzer, TemporalResidualMonitor
from subsystems.map.monitoring.level1 import FORMULA, frozen_residual_calibration, level1_scores
from subsystems.map.monitoring.level2 import (
    Level2Parameters,
    Level2Result,
    compute_level2,
    frozen_level2_calibration,
    level2_metadata,
)
from subsystems.map.monitoring.level3 import (
    Level3Parameters,
    Level3Result,
    compute_level3,
    level3_metadata,
)
from subsystems.map.monitoring.spatial_coherence import SpatialCoherenceRegion
from subsystems.map.monitoring.series_export import (
    build_records,
    field_units,
    phases,
)
from subsystems.map.pipelines.learning_pipeline import LearningPipeline
from subsystems.map.utils.config_loader import load_config
from subsystems.map.utils.contract import CONTRACT_VERSION, SCHEMA_VERSION
from subsystems.map.utils.forecast_timing import validate_model_target
from subsystems.map.utils.experiment_paths import (
    experiment_model_directory,
    results_directory,
    static_file_path,
)
from subsystems.map.utils.temporal_windows import (
    TemporalWindow,
    resolve_temporal_window,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitoringFrame:
    """Monitoring evidence available at one acquisition date."""

    index: int
    date: str
    observed_los: float
    predicted_los: float
    prediction_spread: float
    residual: float
    velocity: float
    acceleration: float
    acceleration_cusum_instability: float
    acceleration_cusum_recovery: float
    l2_z: float
    l2_instability_cusum: float
    l2_overforecast_cusum: float
    l2_alarm: bool
    regime_change_score: float
    dynamics: str
    l3_level: str


@dataclass(frozen=True)
class SimulationResult:
    """Inputs and causal frame results needed by exporters."""

    dates: tuple[str, ...]
    calibration: TemporalWindow
    monitoring: TemporalWindow
    observed: np.ndarray
    predicted: np.ndarray
    uncertainty: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    acceleration_cusum_instability: np.ndarray
    acceleration_cusum_recovery: np.ndarray
    regional_acceleration_cusum_instability: np.ndarray
    regional_acceleration_cusum_recovery: np.ndarray
    regional_cusum_available: np.ndarray
    regional_dynamics: np.ndarray
    regime_change_score: np.ndarray
    dynamics: np.ndarray
    frames: tuple[MonitoringFrame, ...]
    acceleration_cusum_decision_threshold: float
    unit: str
    native_unit: str
    value_scale: float
    forecast_origin_dates: tuple[str | None, ...] = ()
    level1: dict[str, Any] | None = None
    level2: Level2Result | None = None
    level3: Level3Result | None = None


def frame_indices(
    dates: tuple[str, ...], calibration: TemporalWindow, monitoring: TemporalWindow
) -> tuple[int, ...]:
    """Return real acquisitions in the configured calibration/monitoring windows."""
    return tuple(
        index
        for index in range(len(dates))
        if calibration.start_index <= index < calibration.end_index
        or monitoring.start_index <= index < monitoring.end_index
    )


def causal_prefix_indices(time_indices: np.ndarray, current_index: int) -> np.ndarray:
    """Select samples known by an acquisition, never samples from its future."""
    return np.flatnonzero(time_indices <= current_index)


def configured_animation_output(config: dict[str, Any]) -> Path:
    """Resolve the scenario-local animation destination from MAP configuration."""
    config_path = Path(str(config['_config_path']))
    animation = config.get('monitoring', {}).get('animation', {})
    if not isinstance(animation, dict):
        raise ValueError('monitoring.animation must be a mapping.')
    directory = str(animation.get('output_directory', 'monitoring/animation'))
    filename = str(animation.get('filename', 'tsf_monitoring.mp4'))
    return results_directory(config, config_path) / directory / filename


def run_simulation(config: dict[str, Any], *, train: bool = True) -> SimulationResult:
    """Train the configured baseline and replay monitoring causally."""
    if train:
        LOGGER.info('Calibrating configured MAP baseline model')
        LearningPipeline(config).run()

    dataset, model = _load_dataset_and_model(config)
    dataset_config = _dataset_config(config)
    calibration = resolve_temporal_window(
        dataset.dates, dataset_config, 'calibration', end_inclusive=False
    )
    monitoring = resolve_temporal_window(dataset.dates, dataset_config, 'monitoring')
    if calibration.end_index > monitoring.start_index:
        raise ValueError('Calibration and monitoring windows must not overlap.')

    analyzer = ResidualAnalyzer()
    observed_stack = analyzer.restore_stack(dataset, dataset.targets)
    prediction_values = np.full(dataset.targets.shape, np.nan, dtype=np.float64)
    uncertainty_values = np.full(dataset.targets.shape, np.nan, dtype=np.float64)

    # Each call sees one acquisition's causal feature rows. There is no batch that
    # contains a later monitoring acquisition.
    for index in range(calibration.start_index, monitoring.end_index):
        rows = np.flatnonzero(dataset.time_indices == index)
        if rows.size == 0:
            continue
        prediction = model.predict(dataset.features[rows])
        prediction_values[rows] = prediction.y_pred
        if prediction.uncertainty is not None:
            uncertainty_values[rows] = prediction.uncertainty

    prediction_stack = analyzer.restore_stack(dataset, prediction_values)
    uncertainty_stack = analyzer.restore_stack(dataset, uncertainty_values)
    observed = _spatial_mean(observed_stack)
    level1 = _simulation_level1(
        config, observed_stack - prediction_stack, dataset.dates,
        calibration, monitoring, dataset.mask, dataset.target_feature,
    )
    level2 = _simulation_level2(
        config, observed_stack - prediction_stack, dataset.dates,
        calibration, monitoring, dataset.mask, dataset.target_feature,
    )
    predicted = np.full(len(dataset.dates), np.nan)
    uncertainty = np.full(len(dataset.dates), np.nan)
    velocity = np.full(len(dataset.dates), np.nan)
    acceleration = np.full(len(dataset.dates), np.nan)
    acceleration_cusum_instability = np.full(len(dataset.dates), np.nan)
    acceleration_cusum_recovery = np.full(len(dataset.dates), np.nan)
    scores = np.full(len(dataset.dates), np.nan)
    regional_scores = np.full(len(dataset.dates), np.nan)
    dynamics = np.full(len(dataset.dates), 'stable', dtype='<U12')
    monitor = TemporalResidualMonitor(config['monitoring']['dashboard'])
    frames: list[MonitoringFrame] = []

    for current in range(monitoring.start_index, monitoring.end_index):
        end = current + 1
        result = monitor.analyze(
            observed_stack[:end],
            prediction_stack[:end],
            dataset.dates[:end],
            (calibration.start_index, calibration.end_index),
            (monitoring.start_index, end),
            uncertainty_stack[:end],
        )
        predicted[current] = result.predicted_mean[current]
        uncertainty[current] = (
            np.nan
            if result.uncertainty_mean is None
            else result.uncertainty_mean[current]
        )
        velocity[current] = result.velocity[current]
        acceleration[current] = result.acceleration[current]
        acceleration_cusum_instability[current] = result.acceleration_cusum_instability[current]
        acceleration_cusum_recovery[current] = result.acceleration_cusum_recovery[current]
        scores[current] = result.regime_change_score[current]
        dynamics[current] = result.dynamics[current]
        frames.append(
            MonitoringFrame(
                index=current,
                date=dataset.dates[current],
                observed_los=float(observed[current]),
                predicted_los=float(predicted[current]),
                prediction_spread=float(uncertainty[current]),
                residual=float(result.residual_mean[current]),
                velocity=float(velocity[current]),
                acceleration=float(acceleration[current]),
                acceleration_cusum_instability=float(acceleration_cusum_instability[current]),
                acceleration_cusum_recovery=float(acceleration_cusum_recovery[current]),
                **_level2_frame(level2, current),
                regime_change_score=float(scores[current]),
                dynamics=str(dynamics[current]),
                l3_level='',
            )
        )

    level3 = _simulation_level3(config, scores, regional_scores, level2, monitoring)
    return SimulationResult(
        dates=dataset.dates,
        forecast_origin_dates=dataset.forecast_origin_dates,
        level1=level1,
        level2=level2,
        level3=level3,
        calibration=calibration,
        monitoring=monitoring,
        observed=observed,
        predicted=predicted,
        uncertainty=uncertainty,
        velocity=velocity,
        acceleration=acceleration,
        acceleration_cusum_instability=acceleration_cusum_instability,
        acceleration_cusum_recovery=acceleration_cusum_recovery,
        regional_acceleration_cusum_instability=np.full(len(dataset.dates), np.nan),
        regional_acceleration_cusum_recovery=np.full(len(dataset.dates), np.nan),
        regional_cusum_available=np.zeros(len(dataset.dates), dtype=bool),
        regional_dynamics=np.full(len(dataset.dates), 'stable', dtype='<U12'),
        regime_change_score=scores,
        dynamics=dynamics,
        frames=_with_l3_levels(frames, level3),
        acceleration_cusum_decision_threshold=monitor.cusum_decision,
        unit=str(config.get('plotting', {}).get('deformation_unit', '')),
        native_unit=str(
            config.get('plotting', {}).get('native_deformation_rate_unit', '')
        ),
        value_scale=float(config.get('plotting', {}).get('value_scale', 1.0)),
    )


def run_precomputed_simulation(
    config: dict[str, Any],
    *,
    dates: tuple[str, ...],
    calibration: TemporalWindow,
    monitoring: TemporalWindow,
    observed_stack: np.ndarray,
    prediction_stack: np.ndarray,
    uncertainty_stack: np.ndarray | None,
    fixed_support_mask: np.ndarray | None,
    coherent_regions: tuple[SpatialCoherenceRegion, ...] = (),
    forecast_origin_dates: tuple[str | None, ...] = (),
    level1: dict[str, Any] | None = None,
    level2: Level2Result | None = None,
    level3: Level3Result | None = None,
) -> SimulationResult:
    """Replay monitoring from persisted inference stacks.

    This is the animation-side counterpart to the independent monitoring
    pipeline. It deliberately does not reload a model or engineered features:
    non-overlapping temporal windows reveal data acquisition by acquisition.
    Overlapping spatial-calibration windows use the fixed full-period baseline
    and therefore produce a retrospective replay.
    """
    observed = _spatial_mean(observed_stack)
    if level1 is None and 'anomaly_detection' in config:
        raise ValueError(
            'Precomputed L1 replay requires the frozen level1 context from '
            'MonitoringPipeline; do not recalibrate from a plotted support.'
        )
    if level2 is None and 'level2' in config:
        raise ValueError(
            'Precomputed L2 replay requires the Level2Result from MonitoringPipeline.'
        )
    monitor = TemporalResidualMonitor(config['monitoring']['dashboard'])
    time_count = len(dates)
    predicted = np.full(time_count, np.nan)
    uncertainty = np.full(time_count, np.nan)
    velocity = np.full(time_count, np.nan)
    acceleration = np.full(time_count, np.nan)
    acceleration_cusum_instability = np.full(time_count, np.nan)
    acceleration_cusum_recovery = np.full(time_count, np.nan)
    regional_acceleration_cusum_instability = np.full(time_count, np.nan)
    regional_acceleration_cusum_recovery = np.full(time_count, np.nan)
    regional_cusum_available = np.zeros(time_count, dtype=bool)
    regional_dynamics = np.full(time_count, 'stable', dtype='<U12')
    scores = np.full(time_count, np.nan)
    regional_scores = np.full(time_count, np.nan)
    dynamics = np.full(time_count, 'stable', dtype='<U12')
    frames: list[MonitoringFrame] = []

    retrospective = calibration.end_index > monitoring.start_index
    fixed_result = None
    if retrospective:
        LOGGER.info(
            'Calibration and monitoring overlap; animation replays one full-period '
            'analysis (L1/L2/L3 baselines still use pre-monitoring dates only).'
        )
        fixed_result = monitor.analyze(
            observed_stack,
            prediction_stack,
            dates,
            (calibration.start_index, calibration.end_index),
            (monitoring.start_index, monitoring.end_index),
            uncertainty_stack,
            fixed_support_mask=fixed_support_mask,
            coherent_regions=coherent_regions,
        )

    for current in range(monitoring.start_index, monitoring.end_index):
        end = current + 1
        result = fixed_result
        if result is None:
            result = monitor.analyze(
                observed_stack[:end],
                prediction_stack[:end],
                dates[:end],
                (calibration.start_index, calibration.end_index),
                (monitoring.start_index, end),
                None if uncertainty_stack is None else uncertainty_stack[:end],
                fixed_support_mask=fixed_support_mask,
                coherent_regions=coherent_regions,
            )
        predicted[current] = result.predicted_mean[current]
        uncertainty[current] = (
            np.nan if result.uncertainty_mean is None else result.uncertainty_mean[current]
        )
        velocity[current] = result.velocity[current]
        acceleration[current] = result.acceleration[current]
        acceleration_cusum_instability[current] = result.acceleration_cusum_instability[current]
        acceleration_cusum_recovery[current] = result.acceleration_cusum_recovery[current]
        regional_acceleration_cusum_instability[current] = result.regional_acceleration_cusum_instability[current]
        regional_acceleration_cusum_recovery[current] = result.regional_acceleration_cusum_recovery[current]
        regional_cusum_available[current] = result.regional_cusum_available[current]
        regional_dynamics[current] = result.regional_dynamics[current]
        scores[current] = result.regime_change_score[current]
        regional_scores[current] = result.regional_regime_change_score[current]
        dynamics[current] = result.dynamics[current]
        frames.append(
            MonitoringFrame(
                index=current,
                date=dates[current],
                observed_los=float(observed[current]),
                predicted_los=float(predicted[current]),
                prediction_spread=float(uncertainty[current]),
                residual=float(result.residual_mean[current]),
                velocity=float(velocity[current]),
                acceleration=float(acceleration[current]),
                acceleration_cusum_instability=float(acceleration_cusum_instability[current]),
                acceleration_cusum_recovery=float(acceleration_cusum_recovery[current]),
                **_level2_frame(level2, current),
                regime_change_score=float(scores[current]),
                dynamics=str(dynamics[current]),
                l3_level='',
            )
        )
    if level3 is None:
        level3 = _simulation_level3(
            config, scores, regional_scores, level2, monitoring, coherent_regions
        )
    return SimulationResult(
        dates=dates,
        forecast_origin_dates=forecast_origin_dates,
        level1=level1,
        level2=level2,
        level3=level3,
        calibration=calibration,
        monitoring=monitoring,
        observed=observed,
        predicted=predicted,
        uncertainty=uncertainty,
        velocity=velocity,
        acceleration=acceleration,
        acceleration_cusum_instability=acceleration_cusum_instability,
        acceleration_cusum_recovery=acceleration_cusum_recovery,
        regional_acceleration_cusum_instability=regional_acceleration_cusum_instability,
        regional_acceleration_cusum_recovery=regional_acceleration_cusum_recovery,
        regional_cusum_available=regional_cusum_available,
        regional_dynamics=regional_dynamics,
        regime_change_score=scores,
        dynamics=dynamics,
        frames=_with_l3_levels(frames, level3),
        acceleration_cusum_decision_threshold=monitor.cusum_decision,
        unit=str(config.get('plotting', {}).get('deformation_unit', '')),
        native_unit=str(config.get('plotting', {}).get('native_deformation_rate_unit', '')),
        value_scale=float(config.get('plotting', {}).get('value_scale', 1.0)),
    )


def _simulation_level1(
    config: dict[str, Any], residual_stack: np.ndarray, dates: tuple[str, ...],
    calibration: TemporalWindow, monitoring: TemporalWindow, mask: np.ndarray,
    target_feature: str,
) -> dict[str, Any]:
    """Standalone replays use the same frozen artifact as MonitoringPipeline."""
    settings = config['anomaly_detection']
    path = Path(str(settings.get('calibration_file', 'monitoring/l1_calibration.json')))
    if not path.is_absolute():
        path = results_directory(config, Path(str(config['_config_path']))) / path
    calibrated = frozen_residual_calibration(
        path, residual_stack, dates, (calibration.start_index, calibration.end_index),
        monitoring.start_index, np.asarray(mask, dtype=bool),
        unit=str(config.get('plotting', {}).get('native_deformation_rate_unit', '')),
        target_feature=target_feature,
    )
    threshold = settings.get('zscore_threshold', 3.0)
    direction = str(settings.get('direction', 'both')).lower()
    level1_scores(np.array([]), calibrated['sigma'], threshold, direction)
    return {
        'formula': FORMULA, 'threshold': float(threshold), 'direction': direction,
        'comparison': 'strict', 'calibration': calibrated,
    }


def _simulation_level2(
    config: dict[str, Any], residual_stack: np.ndarray, dates: tuple[str, ...],
    calibration: TemporalWindow, monitoring: TemporalWindow, mask: np.ndarray,
    target_feature: str,
) -> Level2Result:
    """Standalone replays share the frozen L2 artifact with MonitoringPipeline."""
    parameters = Level2Parameters.from_config(config)
    path = Path(str(config.get('level2', {}).get(
        'calibration_file', 'monitoring/l2_calibration.json'
    )))
    if not path.is_absolute():
        path = results_directory(config, Path(str(config['_config_path']))) / path
    windows = (calibration.start_index, calibration.end_index)
    mask = np.asarray(mask, dtype=bool)
    calibrated = frozen_level2_calibration(
        path, residual_stack, dates, windows, monitoring.start_index, mask,
        unit=str(config.get('plotting', {}).get('native_deformation_rate_unit', '')),
        target_feature=target_feature, parameters=parameters,
    )
    return compute_level2(
        residual_stack, mask, windows,
        (monitoring.start_index, monitoring.end_index), calibrated, parameters,
    )


def _simulation_level3(
    config: dict[str, Any],
    scores: np.ndarray,
    regional_scores: np.ndarray,
    level2: Level2Result | None,
    monitoring: TemporalWindow,
    regions: tuple[SpatialCoherenceRegion, ...] = (),
) -> Level3Result | None:
    """Apply the L3 evidence rule to replayed scores; it needs L2 alarms."""
    if level2 is None:
        return None
    return compute_level3(
        scores, regional_scores, level2,
        (monitoring.start_index, monitoring.end_index),
        Level3Parameters.from_config(config), regions,
    )


def _with_l3_levels(
    frames: list[MonitoringFrame], level3: Level3Result | None
) -> tuple[MonitoringFrame, ...]:
    """Attach the L3 level, which needs the whole causal score history."""
    return tuple(
        replace(frame, l3_level=_l3_level(level3, frame.index)) for frame in frames
    )


def _l3_level(level3: Level3Result | None, index: int) -> str:
    return 'n/a' if level3 is None else str(level3.level[index])


def _level2_frame(level2: Level2Result | None, index: int) -> dict[str, Any]:
    """L2 values of one acquisition; NaN/false when L2 is not configured."""
    if level2 is None:
        return {
            'l2_z': np.nan, 'l2_instability_cusum': np.nan,
            'l2_overforecast_cusum': np.nan, 'l2_alarm': False,
        }
    return {
        'l2_z': float(level2.z[index]),
        'l2_instability_cusum': float(level2.instability_cusum[index]),
        'l2_overforecast_cusum': float(level2.overforecast_cusum[index]),
        'l2_alarm': bool(level2.alarm[index]),
    }


def _level2_series(level2: Level2Result | None, name: str, size: int) -> np.ndarray:
    """Full L2 series by attribute name, or NaN when L2 is not configured."""
    if level2 is None:
        return np.full(size, False if name.endswith('alarm') else np.nan)
    return getattr(level2, name)


def _canonical_columns(result: SimulationResult) -> dict[str, Any]:
    """Animation series under contract v1.0 names, rates in display units."""
    size = len(result.dates)
    scale = result.value_scale
    origins = result.forecast_origin_dates or ('',) * size
    columns: dict[str, Any] = {
        'date': result.dates,
        'forecast_origin_date': [origin or None for origin in origins],
        'phase': phases(
            size,
            (result.calibration.start_index, result.calibration.end_index),
            (result.monitoring.start_index, result.monitoring.end_index),
        ),
        'observed_mean_los_velocity': result.observed * scale,
        'predicted_baseline_velocity': result.predicted * scale,
        'prediction_spread': result.uncertainty * scale,
        'mean_residual_rate': (result.observed - result.predicted) * scale,
        'acceleration_cusum_input': result.acceleration * scale,
        'acceleration_cusum_instability': result.acceleration_cusum_instability,
        'acceleration_cusum_recovery': result.acceleration_cusum_recovery,
        'dynamics': result.dynamics,
        'coherent_region_available': result.regional_cusum_available,
        'coherent_region_acceleration_cusum_instability': (
            result.regional_acceleration_cusum_instability
        ),
        'coherent_region_acceleration_cusum_recovery': (
            result.regional_acceleration_cusum_recovery
        ),
        'coherent_region_dynamics': np.where(
            result.regional_cusum_available, result.regional_dynamics, ''
        ),
        'regime_change_score': result.regime_change_score,
    }
    if result.level2 is not None:
        level2 = result.level2
        columns.update(
            {
                'l2_z': level2.z,
                'l2_instability_cusum': level2.instability_cusum,
                'l2_overforecast_cusum': level2.overforecast_cusum,
                'l2_alarm': level2.alarm,
                'coherent_region_l2_instability_cusum': (
                    level2.regional_instability_cusum
                ),
                'coherent_region_l2_overforecast_cusum': (
                    level2.regional_overforecast_cusum
                ),
                'coherent_region_l2_alarm': level2.regional_alarm,
            }
        )
    if result.level3 is not None:
        columns.update(
            {
                'coherent_region_regime_change_score': result.level3.regional_score,
                'l3_level': result.level3.level,
                **{
                    f'l3_{name}': flags
                    for name, flags in result.level3.criteria.items()
                },
            }
        )
    return columns


def export_csv(result: SimulationResult, output: Path) -> Path:
    """Write the display-scaled causal trajectory alongside the visual product.

    Column names follow contract v1.0; units are listed in the graph JSON.
    """
    path = output.with_suffix('.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(
        _canonical_columns(result), [frame.index for frame in result.frames]
    )
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else [])
        writer.writeheader()
        writer.writerows(
            {key: '' if value is None else value for key, value in row.items()}
            for row in records
        )
    return path


def export_dashboard_json(result: SimulationResult, output: Path) -> Path:
    """Write all animation-dashboard graph inputs as web-ready JSON.

    Field names follow contract v1.0 (shared with ``monitoring_series``).
    Rates are converted to the displayed unit; missing values are ``null``.
    """
    path = output.with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(
        _canonical_columns(result),
        range(result.calibration.start_index, result.monitoring.end_index),
    )
    payload = {
        'schema_version': SCHEMA_VERSION,
        'contract_version': CONTRACT_VERSION,
        'product': 'map_tsf_monitoring_animation_graph_data',
        'units': {
            'deformation_rate': result.unit,
            'acceleration': _acceleration_unit(result.unit),
            **field_units(list(records[0]) if records else [], result.unit),
        },
        'temporal_windows': {
            'calibration': _window_payload(result.calibration),
            'monitoring': _window_payload(result.monitoring),
        },
        'thresholds': {
            'acceleration_cusum_decision_threshold': (
                result.acceleration_cusum_decision_threshold
            ),
            'l2_decision_threshold': (
                None
                if result.level2 is None
                else result.level2.parameters.decision_threshold
            ),
            'l3_watch_threshold': (
                None if result.level3 is None else result.level3.parameters.watch_threshold
            ),
            'l3_alert_threshold': (
                None if result.level3 is None else result.level3.parameters.alert_threshold
            ),
            'l3_consecutive_acquisitions': (
                None
                if result.level3 is None
                else result.level3.parameters.consecutive_acquisitions
            ),
        },
        'series': records,
    }
    if result.level1 is not None:
        # Residuals in graph exports use display units; sigma must use the same
        # scale. Threshold, direction and the calibration population are shared
        # with maps, rather than re-estimated from an aggregate plotted series.
        calibration = dict(result.level1['calibration'])
        calibration.update(
            native_sigma=calibration['sigma'],
            native_unit=calibration['unit'],
            sigma=calibration['sigma'] * result.value_scale,
            unit=result.unit,
            mean_residual_diagnostic=(
                calibration['mean_residual_diagnostic'] * result.value_scale
            ),
        )
        payload['level1'] = {**result.level1, 'calibration': calibration}
    if result.level2 is not None:
        # Same display-unit convention as L1: z and CUSUM are unit-free.
        level2 = level2_metadata(result.level2, result.dates)
        calibration = dict(level2['calibration'])
        calibration.update(
            native_sigma=calibration['sigma'],
            native_unit=calibration['unit'],
            sigma=calibration['sigma'] * result.value_scale,
            unit=result.unit,
            mean_residual=calibration['mean_residual'] * result.value_scale,
        )
        regional = [
            {
                **entry,
                'sigma': entry['sigma'] * result.value_scale,
                'mean_residual': entry['mean_residual'] * result.value_scale,
                'unit': result.unit,
            }
            for entry in level2['regional_calibration']
        ]
        payload['level2'] = {
            **level2, 'calibration': calibration, 'regional_calibration': regional,
        }
    if result.level3 is not None:
        level3 = level3_metadata(result.level3, result.dates)
        # The per-date L3 series is already merged into ``series``.
        level3.pop('series')
        payload['level3'] = level3
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def _window_payload(window: TemporalWindow) -> dict[str, int | str]:
    """Return a web-friendly description of one inclusive/exclusive window."""
    return {
        'start_index': window.start_index,
        'end_index_exclusive': window.end_index,
        'configured_start_date': window.start_date,
        'configured_end_date': window.end_date,
    }


def save_visualization(
    result: SimulationResult,
    output: Path,
    *,
    fps: int,
    dpi: int,
    show: bool,
) -> Path:
    """Save PNG, GIF, or MP4, falling back from MP4 to Pillow GIF."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, update = _create_figure(result)
    indices = frame_indices(result.dates, result.calibration, result.monitoring)
    if output.suffix.lower() == '.png':
        update(indices[-1])
        figure.savefig(output, dpi=dpi, bbox_inches='tight')
    else:
        animation = FuncAnimation(
            figure, update, frames=indices, interval=1000 / fps, blit=False
        )
        if output.suffix.lower() == '.mp4':
            if FFMpegWriter.isAvailable():
                animation.save(output, writer=FFMpegWriter(fps=fps), dpi=dpi)
            else:
                output = output.with_suffix('.gif')
                LOGGER.warning('ffmpeg unavailable; writing %s with Pillow', output)
                animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
        elif output.suffix.lower() == '.gif':
            animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
        else:
            raise ValueError('Output extension must be .png, .mp4, or .gif.')
    if show:
        plt.show()
    plt.close(figure)
    return output


def _create_figure(result: SimulationResult) -> tuple[Any, Any]:
    """Create the scientific four-panel figure and frame updater."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = np.array([datetime.fromisoformat(value) for value in result.dates])
    scale = result.value_scale
    rate_unit = result.unit or 'native rate'
    figure, axes = plt.subplots(4, 1, figsize=(12, 11.5), sharex=True)
    figure.subplots_adjust(top=0.88, hspace=0.25)
    boundary = datetime.fromisoformat(result.monitoring.start_date)
    calibration_start = datetime.fromisoformat(result.calibration.start_date)
    monitoring_end = dates[result.monitoring.end_index - 1]
    for axis in axes:
        axis.axvspan(calibration_start, boundary, color='#4c78a8', alpha=0.08)
        axis.axvspan(boundary, monitoring_end, color='#f58518', alpha=0.06)
        axis.axvline(
            boundary,
            color='0.78',
            linestyle='--',
            linewidth=0.8,
            zorder=1,
        )
        axis.grid(alpha=0.22)
    axes[0].set_title('A. LOS deformation rate / baseline behaviour', loc='left')
    axes[1].set_title('B. L2: CUSUM of standardized residuals', loc='left')
    axes[2].set_title('C. Supporting evidence: acceleration CUSUM', loc='left')
    axes[3].set_title(
        'D. L3: regime-change score (heuristic, not a probability)', loc='left'
    )
    axes[0].set_ylabel(f'LOS deformation rate [{rate_unit}]')
    axes[1].set_ylabel('L2 CUSUM [sigma, symlog]')
    axes[2].set_ylabel('Acceleration CUSUM')
    axes[3].set_ylabel('Score [0–1]')
    axes[3].set_ylim(-0.03, 1.03)
    size = len(result.dates)
    l2_instability = _level2_series(result.level2, 'instability_cusum', size)
    l2_overforecast = _level2_series(result.level2, 'overforecast_cusum', size)
    l2_regional = _level2_series(result.level2, 'regional_instability_cusum', size)
    l2_regional_available = (
        np.zeros(size, dtype=bool)
        if result.level2 is None
        else result.level2.regional_available
    )
    l2_threshold = (
        None if result.level2 is None else result.level2.parameters.decision_threshold
    )
    l2_upper = max(
        1.0,
        l2_threshold or 0.0,
        _finite_max(l2_instability),
        _finite_max(l2_overforecast),
        _finite_max(l2_regional),
    )
    if l2_threshold is not None:
        # Linear up to h keeps the decision level readable past large alarms.
        axes[1].set_yscale('symlog', linthresh=l2_threshold)
        axes[1].set_ylim(0.0, 1.3 * l2_upper)
    else:
        axes[1].set_ylim(-0.05 * l2_upper, 1.05 * l2_upper)
    if l2_threshold is not None:
        axes[1].axhline(
            l2_threshold, color='black', linestyle='--', linewidth=1.2,
            label=f'L2 decision threshold h ({l2_threshold:g})',
        )
    (l2_instability_line,) = axes[1].plot(
        [], [], color='#d62728', linewidth=1.8, label='TSF-wide L2 instability CUSUM'
    )
    (l2_overforecast_line,) = axes[1].plot(
        [], [], color='0.55', linewidth=0.9, linestyle='--',
        label='Over-forecast CUSUM (diagnostic)',
    )
    (l2_regional_line,) = axes[1].plot(
        [], [], color='darkorchid', linewidth=1.5,
        label='Coherent-region L2 instability CUSUM',
    )
    # A monitoring replay must use one coordinate frame for every acquisition.
    # Otherwise Matplotlib rescales early frames to a handful of observations,
    # making ordinary variation look like a pronounced trend or deceleration.
    for axis in axes:
        axis.set_xlim(dates[0], monitoring_end)
    axes[0].set_ylim(
        *_padded_limits(result.observed * scale, result.predicted * scale),
    )
    cusum_upper = max(
        1.0,
        result.acceleration_cusum_decision_threshold,
        _finite_max(result.acceleration_cusum_instability),
        _finite_max(result.acceleration_cusum_recovery),
    )
    axes[2].set_ylim(-0.05 * cusum_upper, 1.05 * cusum_upper)
    regional_axis = axes[2].twinx()
    regional_upper = max(
        1.0,
        _finite_max(result.regional_acceleration_cusum_instability),
        _finite_max(result.regional_acceleration_cusum_recovery),
    )
    regional_axis.set_ylim(-0.05 * regional_upper, 1.05 * regional_upper)
    regional_axis.set_ylabel('Coherent-region CUSUM statistic')
    if result.level3 is not None:
        parameters = result.level3.parameters
        axes[3].axhline(
            parameters.watch_threshold,
            color='#e6a700',
            linestyle='--',
            label='Watch score threshold',
        )
        axes[3].axhline(
            parameters.alert_threshold,
            color='#c62828',
            linestyle='--',
            label='Alert score threshold',
        )
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator())
    locator = axes[3].xaxis.get_major_locator()
    axes[3].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    (observed_line,) = axes[0].plot(
        [],
        [],
        color='0.62',
        marker='o',
        markersize=3,
        linewidth=1,
        label='Historical observed rate',
    )
    (current_observation,) = axes[0].plot(
        [],
        [],
        color='black',
        marker='o',
        markersize=4,
        linestyle='none',
        label='Current observation',
        zorder=5,
    )
    (predicted_line,) = axes[0].plot(
        [], [], color='#4c78a8', label='Predicted baseline rate'
    )
    (deceleration_cusum_line,) = axes[2].plot(
        [], [], color='#54a24b', linewidth=0.9, linestyle='--', alpha=0.75,
        label='TSF-wide acceleration CUSUM (recovery side)'
    )
    (acceleration_cusum_line,) = axes[2].plot(
        [], [], color='#e45756', linewidth=0.9, linestyle='--', alpha=0.75,
        label='TSF-wide acceleration CUSUM (instability side)'
    )
    axes[2].axhline(
        result.acceleration_cusum_decision_threshold,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label=(
            'Acceleration decision threshold '
            f'({result.acceleration_cusum_decision_threshold:g})'
        ),
    )
    (regional_acceleration_line,) = regional_axis.plot(
        [], [], color='darkorchid', linewidth=1.5,
        label='Coherent-region acceleration CUSUM (instability side)',
    )
    (regional_deceleration_line,) = regional_axis.plot(
        [], [], color='teal', linewidth=1.5,
        label='Coherent-region acceleration CUSUM (recovery side)',
    )
    (regional_acceleration_period,) = axes[2].plot(
        [], [], color='firebrick', linewidth=4.0,
        transform=axes[2].get_xaxis_transform(),
        label='Regional acceleration period',
    )
    (regional_deceleration_period,) = axes[2].plot(
        [], [], color='seagreen', linewidth=4.0,
        transform=axes[2].get_xaxis_transform(),
        label='Regional deceleration / recovery period',
    )
    (score_line,) = axes[3].plot(
        [], [], color='#7b2cbf', linewidth=2, label='Regime-change score'
    )
    cursors = [
        axis.axvline(
            dates[0],
            color='0.78',
            linestyle='--',
            linewidth=0.8,
            zorder=1,
        )
        for axis in axes
    ]
    status = figure.text(0.5, 0.965, '', ha='center', va='top', family='monospace')
    figure.text(0.25, 0.905, 'CALIBRATION', ha='center', color='#345b83')
    figure.text(0.73, 0.905, 'MONITORING', ha='center', color='#a85500')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[1].legend(loc='upper left', fontsize=8, ncols=2)
    left_handles, left_labels = axes[2].get_legend_handles_labels()
    right_handles, right_labels = regional_axis.get_legend_handles_labels()
    axes[2].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc='upper left', fontsize=8, ncols=2,
    )
    axes[3].legend(loc='upper left', fontsize=8)

    def update(current: int) -> tuple[Any, ...]:
        visible = np.arange(len(dates)) <= current
        historical = np.arange(len(dates)) < current
        observed_line.set_data(dates[historical], result.observed[historical] * scale)
        current_observation.set_data(
            [dates[current]], [result.observed[current] * scale]
        )
        monitoring_visible = visible & (
            np.arange(len(dates)) >= result.monitoring.start_index
        )
        predicted_line.set_data(
            dates[monitoring_visible], result.predicted[monitoring_visible] * scale
        )
        l2_instability_line.set_data(
            dates[monitoring_visible], l2_instability[monitoring_visible]
        )
        l2_overforecast_line.set_data(
            dates[monitoring_visible], l2_overforecast[monitoring_visible]
        )
        l2_regional_visible = monitoring_visible & l2_regional_available
        l2_regional_line.set_data(
            dates[l2_regional_visible], l2_regional[l2_regional_visible]
        )
        acceleration_cusum_line.set_data(
            dates[monitoring_visible],
            result.acceleration_cusum_instability[monitoring_visible],
        )
        deceleration_cusum_line.set_data(
            dates[monitoring_visible],
            result.acceleration_cusum_recovery[monitoring_visible],
        )
        regional_visible = monitoring_visible & result.regional_cusum_available
        regional_acceleration_line.set_data(
            dates[regional_visible],
            result.regional_acceleration_cusum_instability[regional_visible],
        )
        regional_deceleration_line.set_data(
            dates[regional_visible],
            result.regional_acceleration_cusum_recovery[regional_visible],
        )
        acceleration_period = (
            monitoring_visible
            & result.regional_cusum_available
            & (result.regional_dynamics == 'accelerating')
        )
        deceleration_period = (
            monitoring_visible
            & result.regional_cusum_available
            & (result.regional_dynamics == 'decelerating')
        )
        regional_acceleration_period.set_data(
            dates[: current + 1],
            np.where(acceleration_period[: current + 1], 0.90, np.nan),
        )
        regional_deceleration_period.set_data(
            dates[: current + 1],
            np.where(deceleration_period[: current + 1], 0.90, np.nan),
        )
        score_line.set_data(
            dates[monitoring_visible], result.regime_change_score[monitoring_visible]
        )
        for cursor in cursors:
            cursor.set_xdata([dates[current], dates[current]])
        if current < result.monitoring.start_index:
            status.set_text(
                f'Current date: {result.dates[current]}  |  Phase: Calibration'
            )
        else:
            score = result.regime_change_score[current]
            l2_status = (
                'n/a'
                if result.level2 is None
                else 'ALARM' if result.level2.alarm[current] else 'normal'
            )
            status.set_text(
                f'Current date: {result.dates[current]}  |  Phase: Monitoring  |  '
                f'L2: {l2_status}  |  '
                f'Dynamics: {result.dynamics[current].title()}  |  '
                f'Regime-change score: {score:.2f}  |  '
                f'L3: {_l3_level(result.level3, current).upper()}'
            )
        return (
            observed_line,
            current_observation,
            predicted_line,
            l2_instability_line,
            l2_overforecast_line,
            l2_regional_line,
            acceleration_cusum_line,
            deceleration_cusum_line,
            regional_acceleration_line,
            regional_deceleration_line,
            regional_acceleration_period,
            regional_deceleration_period,
            score_line,
            *cursors,
            status,
        )

    return figure, update


def _padded_limits(*values: np.ndarray) -> tuple[float, float]:
    """Return finite global vertical limits with a small visual margin."""
    finite = np.concatenate(
        [np.asarray(value, dtype=float)[np.isfinite(value)] for value in values]
    )
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    spread = upper - lower
    padding = max(0.001, spread * 0.08)
    return lower - padding, upper + padding


def _finite_max(values: np.ndarray) -> float:
    """Return a safe non-negative maximum for a plotted monitoring series."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return 0.0 if finite.size == 0 else max(0.0, float(np.max(finite)))


def _acceleration_unit(rate_unit: str) -> str:
    """Return the temporal-derivative unit for a displayed deformation rate."""
    if '/day' in rate_unit:
        return rate_unit.replace('/day', '/day²')
    return f'{rate_unit}/day'


def _load_dataset_and_model(config: dict[str, Any]) -> tuple[Dataset, Any]:
    """Load the standard MAP dataset and configured registered model artifact."""
    import subsystems.map.plugins.models  # noqa: F401

    dataset_config = _dataset_config(config)
    names = [str(value) for value in dataset_config['features']]
    target = str(dataset_config['target_feature'])
    config_path = Path(str(config['_config_path']))
    results_root = results_directory(config, config_path)
    feature_paths = [
        results_root / 'features',
        results_root / 'temporal_features',
        results_root / 'meteo_features',
    ]
    loaded = FeatureLoader(
        feature_paths,
        static_file_path(config, config_path, dataset_config['mask_file']),
        str(config['data'].get('temporal_alignment_method', 'exact')),
    ).load(list(dict.fromkeys([*names, target])), reference_feature=target)
    builder = DatasetBuilder()
    dataset = builder.build(loaded, names, target)
    model_name = str(config['model'])
    model_path = (
        experiment_model_directory(results_root, config)
        / 'model.pkl'
    )
    if not model_path.is_file():
        raise FileNotFoundError(f'MAP model artifact not found: {model_path}')
    model = MODEL_REGISTRY[model_name].load(model_path)
    validate_model_target(model_path, target, names)
    specification = model.sequence_spec()
    if specification is not None:
        dataset = builder.build_sequences(dataset, *specification)
    return dataset, model


def _dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get('datasets', {}).get(config.get('dataset'))
    if not isinstance(value, dict):
        raise KeyError('Configured MAP dataset was not found.')
    return value


def _spatial_mean(stack: np.ndarray) -> np.ndarray:
    finite = np.isfinite(stack)
    count = finite.sum(axis=(1, 2))
    return np.divide(
        np.nansum(stack, axis=(1, 2)),
        count,
        out=np.full(stack.shape[0], np.nan),
        where=count > 0,
    )


def main() -> None:
    """Run the configured operational replay and export its products."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--reuse-model',
        action='store_true',
        help='Use an existing calibrated artifact instead of retraining it.',
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    if args.fps < 1 or args.dpi < 1:
        parser.error('--fps and --dpi must be positive.')
    config = load_config(args.config)
    result = run_simulation(config, train=not args.reuse_model)
    visual_path = save_visualization(
        result, args.output, fps=args.fps, dpi=args.dpi, show=args.show
    )
    csv_path = export_csv(result, visual_path)
    json_path = export_dashboard_json(result, visual_path)
    LOGGER.info('Wrote monitoring visualization: %s', visual_path)
    LOGGER.info('Wrote monitoring trajectory: %s', csv_path)
    LOGGER.info('Wrote dashboard graph data: %s', json_path)


if __name__ == '__main__':
    main()
