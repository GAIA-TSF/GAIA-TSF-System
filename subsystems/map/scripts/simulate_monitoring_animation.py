"""Animate causal MAP calibration and acquisition-by-acquisition monitoring.

Usage:
python3 subsystems/map/scripts/simulate_monitoring_animation.py \
    --config subsystems/map/config.yaml \
    --reuse-model \
    --fps 4 \
    --dpi 150

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import Dataset, DatasetBuilder, FeatureLoader
from subsystems.map.monitoring import ResidualAnalyzer, TemporalResidualMonitor
from subsystems.map.monitoring.spatial_coherence import SpatialCoherenceRegion
from subsystems.map.pipelines.learning_pipeline import LearningPipeline
from subsystems.map.utils.config_loader import load_config
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
    prediction_std: float
    residual: float
    velocity: float
    acceleration: float
    acceleration_cusum: float
    deceleration_cusum: float
    regime_change_probability: float
    dynamics: str
    risk_level: str


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
    acceleration_cusum: np.ndarray
    deceleration_cusum: np.ndarray
    regional_acceleration_cusum: np.ndarray
    regional_deceleration_cusum: np.ndarray
    regional_cusum_available: np.ndarray
    regional_dynamics: np.ndarray
    regime_probability: np.ndarray
    dynamics: np.ndarray
    frames: tuple[MonitoringFrame, ...]
    medium_threshold: float
    high_threshold: float
    cusum_decision_threshold: float
    unit: str
    native_unit: str
    value_scale: float


def classify_risk(probability: float, medium: float, high: float) -> str:
    """Classify configured regime-change evidence thresholds."""
    if probability >= high:
        return 'HIGH'
    if probability >= medium:
        return 'MEDIUM'
    return 'NORMAL'


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
    predicted = np.full(len(dataset.dates), np.nan)
    uncertainty = np.full(len(dataset.dates), np.nan)
    velocity = np.full(len(dataset.dates), np.nan)
    acceleration = np.full(len(dataset.dates), np.nan)
    acceleration_cusum = np.full(len(dataset.dates), np.nan)
    deceleration_cusum = np.full(len(dataset.dates), np.nan)
    probabilities = np.full(len(dataset.dates), np.nan)
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
        acceleration_cusum[current] = result.acceleration_cusum[current]
        deceleration_cusum[current] = result.deceleration_cusum[current]
        probabilities[current] = result.regime_risk[current]
        dynamics[current] = result.dynamics[current]
        probability = float(probabilities[current])
        frames.append(
            MonitoringFrame(
                index=current,
                date=dataset.dates[current],
                observed_los=float(observed[current]),
                predicted_los=float(predicted[current]),
                prediction_std=float(uncertainty[current]),
                residual=float(result.residual_mean[current]),
                velocity=float(velocity[current]),
                acceleration=float(acceleration[current]),
                acceleration_cusum=float(acceleration_cusum[current]),
                deceleration_cusum=float(deceleration_cusum[current]),
                regime_change_probability=probability,
                dynamics=str(dynamics[current]),
                risk_level=classify_risk(
                    probability,
                    result.medium_risk_threshold,
                    result.high_risk_threshold,
                ),
            )
        )

    return SimulationResult(
        dates=dataset.dates,
        calibration=calibration,
        monitoring=monitoring,
        observed=observed,
        predicted=predicted,
        uncertainty=uncertainty,
        velocity=velocity,
        acceleration=acceleration,
        acceleration_cusum=acceleration_cusum,
        deceleration_cusum=deceleration_cusum,
        regional_acceleration_cusum=np.full(len(dataset.dates), np.nan),
        regional_deceleration_cusum=np.full(len(dataset.dates), np.nan),
        regional_cusum_available=np.zeros(len(dataset.dates), dtype=bool),
        regional_dynamics=np.full(len(dataset.dates), 'stable', dtype='<U12'),
        regime_probability=probabilities,
        dynamics=dynamics,
        frames=tuple(frames),
        medium_threshold=monitor.medium_risk_threshold,
        high_threshold=monitor.high_risk_threshold,
        cusum_decision_threshold=monitor.cusum_decision,
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
) -> SimulationResult:
    """Replay monitoring causally from persisted inference stacks.

    This is the animation-side counterpart to the independent monitoring
    pipeline. It deliberately does not reload a model or engineered features:
    each animation frame only reveals the subset of already-persisted inference
    products available by that acquisition date.
    """
    observed = _spatial_mean(observed_stack)
    monitor = TemporalResidualMonitor(config['monitoring']['dashboard'])
    time_count = len(dates)
    predicted = np.full(time_count, np.nan)
    uncertainty = np.full(time_count, np.nan)
    velocity = np.full(time_count, np.nan)
    acceleration = np.full(time_count, np.nan)
    acceleration_cusum = np.full(time_count, np.nan)
    deceleration_cusum = np.full(time_count, np.nan)
    regional_acceleration_cusum = np.full(time_count, np.nan)
    regional_deceleration_cusum = np.full(time_count, np.nan)
    regional_cusum_available = np.zeros(time_count, dtype=bool)
    regional_dynamics = np.full(time_count, 'stable', dtype='<U12')
    probabilities = np.full(time_count, np.nan)
    dynamics = np.full(time_count, 'stable', dtype='<U12')
    frames: list[MonitoringFrame] = []

    for current in range(monitoring.start_index, monitoring.end_index):
        end = current + 1
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
        acceleration_cusum[current] = result.acceleration_cusum[current]
        deceleration_cusum[current] = result.deceleration_cusum[current]
        regional_acceleration_cusum[current] = result.regional_acceleration_cusum[current]
        regional_deceleration_cusum[current] = result.regional_deceleration_cusum[current]
        regional_cusum_available[current] = result.regional_cusum_available[current]
        regional_dynamics[current] = result.regional_dynamics[current]
        probabilities[current] = result.regime_risk[current]
        dynamics[current] = result.dynamics[current]
        frames.append(
            MonitoringFrame(
                index=current,
                date=dates[current],
                observed_los=float(observed[current]),
                predicted_los=float(predicted[current]),
                prediction_std=float(uncertainty[current]),
                residual=float(result.residual_mean[current]),
                velocity=float(velocity[current]),
                acceleration=float(acceleration[current]),
                acceleration_cusum=float(acceleration_cusum[current]),
                deceleration_cusum=float(deceleration_cusum[current]),
                regime_change_probability=float(probabilities[current]),
                dynamics=str(dynamics[current]),
                risk_level=classify_risk(
                    float(probabilities[current]),
                    result.medium_risk_threshold,
                    result.high_risk_threshold,
                ),
            )
        )
    return SimulationResult(
        dates=dates,
        calibration=calibration,
        monitoring=monitoring,
        observed=observed,
        predicted=predicted,
        uncertainty=uncertainty,
        velocity=velocity,
        acceleration=acceleration,
        acceleration_cusum=acceleration_cusum,
        deceleration_cusum=deceleration_cusum,
        regional_acceleration_cusum=regional_acceleration_cusum,
        regional_deceleration_cusum=regional_deceleration_cusum,
        regional_cusum_available=regional_cusum_available,
        regional_dynamics=regional_dynamics,
        regime_probability=probabilities,
        dynamics=dynamics,
        frames=tuple(frames),
        medium_threshold=monitor.medium_risk_threshold,
        high_threshold=monitor.high_risk_threshold,
        cusum_decision_threshold=monitor.cusum_decision,
        unit=str(config.get('plotting', {}).get('deformation_unit', '')),
        native_unit=str(config.get('plotting', {}).get('native_deformation_rate_unit', '')),
        value_scale=float(config.get('plotting', {}).get('value_scale', 1.0)),
    )


def export_csv(result: SimulationResult, output: Path) -> Path:
    """Write the display-scaled causal trajectory alongside the visual product."""
    path = output.with_suffix('.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                'date',
                f'observed_rate_[{result.unit}]',
                f'predicted_rate_[{result.unit}]',
                f'prediction_std_[{result.unit}]',
                f'residual_rate_[{result.unit}]',
                f'velocity_[{result.unit}]',
                f'acceleration_[{_acceleration_unit(result.unit)}]',
                'acceleration_cusum_statistic',
                'deceleration_cusum_statistic',
                'regime_change_probability',
                'dynamics',
                'risk_level',
            )
        )
        for frame in result.frames:
            writer.writerow(
                [
                    frame.date,
                    frame.observed_los * result.value_scale,
                    frame.predicted_los * result.value_scale,
                    frame.prediction_std * result.value_scale,
                    frame.residual * result.value_scale,
                    frame.velocity * result.value_scale,
                    frame.acceleration * result.value_scale,
                    frame.acceleration_cusum,
                    frame.deceleration_cusum,
                    frame.regime_change_probability,
                    frame.dynamics,
                    frame.risk_level,
                ]
            )
    return path


def export_dashboard_json(result: SimulationResult, output: Path) -> Path:
    """Write all animation-dashboard graph inputs as web-ready JSON.

    Rate and acceleration values are converted to the displayed unit. Missing
    values are encoded as JSON ``null`` rather than non-standard NaN values.
    CUSUM values and regime scores are dimensionless.
    """
    path = output.with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = range(result.calibration.start_index, result.monitoring.end_index)
    records: list[dict[str, Any]] = []
    for index in indices:
        observed = _json_number(result.observed[index] * result.value_scale)
        predicted = _json_number(result.predicted[index] * result.value_scale)
        records.append(
            {
                'date': result.dates[index],
                'phase': (
                    'calibration'
                    if index < result.monitoring.start_index
                    else 'monitoring'
                ),
                'observed_mean_los_velocity': observed,
                'predicted_baseline_velocity': predicted,
                'prediction_uncertainty': _json_number(
                    result.uncertainty[index] * result.value_scale
                ),
                'mean_residual_rate': (
                    None if observed is None or predicted is None else observed - predicted
                ),
                'observed_acceleration': _json_number(
                    result.acceleration[index] * result.value_scale
                ),
                'tsf_wide_acceleration_cusum': _json_number(
                    result.acceleration_cusum[index]
                ),
                'tsf_wide_deceleration_cusum': _json_number(
                    result.deceleration_cusum[index]
                ),
                'coherent_region_available': bool(
                    result.regional_cusum_available[index]
                ),
                'coherent_region_acceleration_cusum': _json_number(
                    result.regional_acceleration_cusum[index]
                ),
                'coherent_region_deceleration_cusum': _json_number(
                    result.regional_deceleration_cusum[index]
                ),
                'coherent_region_dynamics': str(result.regional_dynamics[index]),
                'regional_acceleration_period': bool(
                    result.regional_cusum_available[index]
                    and result.regional_dynamics[index] == 'accelerating'
                ),
                'regional_deceleration_recovery_period': bool(
                    result.regional_cusum_available[index]
                    and result.regional_dynamics[index] == 'decelerating'
                ),
                'regime_change_score': _json_number(result.regime_probability[index]),
                'tsf_wide_dynamics': str(result.dynamics[index]),
            }
        )
    payload = {
        'schema_version': '1.0',
        'product': 'map_tsf_monitoring_animation_graph_data',
        'units': {
            'deformation_rate': result.unit,
            'acceleration': _acceleration_unit(result.unit),
            'cusum': 'dimensionless',
            'regime_change_score': '[0, 1]',
        },
        'temporal_windows': {
            'calibration': _window_payload(result.calibration),
            'monitoring': _window_payload(result.monitoring),
        },
        'thresholds': {
            'cusum_decision_threshold': result.cusum_decision_threshold,
            'regime_medium_threshold': result.medium_threshold,
            'regime_high_threshold': result.high_threshold,
        },
        'series': records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def _json_number(value: float | np.floating[Any]) -> float | None:
    """Convert a finite numeric value to JSON, otherwise return ``null``."""
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


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
    """Create the scientific three-panel figure and frame updater."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = np.array([datetime.fromisoformat(value) for value in result.dates])
    scale = result.value_scale
    rate_unit = result.unit or 'native rate'
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    figure.subplots_adjust(top=0.88, hspace=0.12)
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
    axes[1].set_title('B. CUSUM early warning', loc='left')
    axes[2].set_title('C. Regime-change probability / risk', loc='left')
    axes[0].set_ylabel(f'LOS deformation rate [{rate_unit}]')
    axes[1].set_ylabel('CUSUM statistic')
    axes[2].set_ylabel('Probability')
    axes[2].set_ylim(-0.03, 1.03)
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
        result.cusum_decision_threshold,
        _finite_max(result.acceleration_cusum),
        _finite_max(result.deceleration_cusum),
    )
    axes[1].set_ylim(-0.05 * cusum_upper, 1.05 * cusum_upper)
    regional_axis = axes[1].twinx()
    regional_upper = max(
        1.0,
        _finite_max(result.regional_acceleration_cusum),
        _finite_max(result.regional_deceleration_cusum),
    )
    regional_axis.set_ylim(-0.05 * regional_upper, 1.05 * regional_upper)
    regional_axis.set_ylabel('Coherent-region CUSUM statistic')
    axes[2].axhline(
        result.medium_threshold,
        color='#e6a700',
        linestyle='--',
        label='Medium threshold',
    )
    axes[2].axhline(
        result.high_threshold, color='#c62828', linestyle='--', label='High threshold'
    )
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())
    locator = axes[2].xaxis.get_major_locator()
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
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
    (deceleration_cusum_line,) = axes[1].plot(
        [], [], color='#54a24b', linewidth=0.9, linestyle='--', alpha=0.75,
        label='TSF-wide deceleration CUSUM'
    )
    (acceleration_cusum_line,) = axes[1].plot(
        [], [], color='#e45756', linewidth=0.9, linestyle='--', alpha=0.75,
        label='TSF-wide acceleration CUSUM'
    )
    axes[1].axhline(
        result.cusum_decision_threshold,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label=f'Decision threshold ({result.cusum_decision_threshold:g})',
    )
    (regional_acceleration_line,) = regional_axis.plot(
        [], [], color='darkorchid', linewidth=1.5,
        label='Coherent-region acceleration CUSUM',
    )
    (regional_deceleration_line,) = regional_axis.plot(
        [], [], color='teal', linewidth=1.5,
        label='Coherent-region deceleration CUSUM',
    )
    (regional_acceleration_period,) = axes[1].plot(
        [], [], color='firebrick', linewidth=4.0,
        transform=axes[1].get_xaxis_transform(),
        label='Regional acceleration period',
    )
    (regional_deceleration_period,) = axes[1].plot(
        [], [], color='seagreen', linewidth=4.0,
        transform=axes[1].get_xaxis_transform(),
        label='Regional deceleration / recovery period',
    )
    (probability_line,) = axes[2].plot(
        [], [], color='#7b2cbf', linewidth=2, label='P(regime change)'
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
    left_handles, left_labels = axes[1].get_legend_handles_labels()
    right_handles, right_labels = regional_axis.get_legend_handles_labels()
    axes[1].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc='upper left', fontsize=8, ncols=2,
    )
    axes[2].legend(loc='upper left', fontsize=8)

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
        acceleration_cusum_line.set_data(
            dates[monitoring_visible],
            result.acceleration_cusum[monitoring_visible],
        )
        deceleration_cusum_line.set_data(
            dates[monitoring_visible],
            result.deceleration_cusum[monitoring_visible],
        )
        regional_visible = monitoring_visible & result.regional_cusum_available
        regional_acceleration_line.set_data(
            dates[regional_visible],
            result.regional_acceleration_cusum[regional_visible],
        )
        regional_deceleration_line.set_data(
            dates[regional_visible],
            result.regional_deceleration_cusum[regional_visible],
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
        probability_line.set_data(
            dates[monitoring_visible], result.regime_probability[monitoring_visible]
        )
        for cursor in cursors:
            cursor.set_xdata([dates[current], dates[current]])
        if current < result.monitoring.start_index:
            status.set_text(
                f'Current date: {result.dates[current]}  |  Phase: Calibration'
            )
        else:
            probability = result.regime_probability[current]
            risk = classify_risk(
                float(probability), result.medium_threshold, result.high_threshold
            )
            status.set_text(
                f'Current date: {result.dates[current]}  |  Phase: Monitoring  |  '
                f'Dynamics: {result.dynamics[current].title()}  |  '
                f'P(regime change): {probability:.2f}  |  Risk level: {risk}'
            )
        return (
            observed_line,
            current_observation,
            predicted_line,
            acceleration_cusum_line,
            deceleration_cusum_line,
            regional_acceleration_line,
            regional_deceleration_line,
            regional_acceleration_period,
            regional_deceleration_period,
            probability_line,
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
