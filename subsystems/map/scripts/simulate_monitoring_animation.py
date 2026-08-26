"""Animate causal MAP calibration and acquisition-by-acquisition monitoring."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
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
from subsystems.map.pipelines.learning_pipeline import LearningPipeline
from subsystems.map.utils.config_loader import load_config
from subsystems.map.utils.experiment_paths import experiment_model_directory
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
    regime_probability: np.ndarray
    dynamics: np.ndarray
    frames: tuple[MonitoringFrame, ...]
    medium_threshold: float
    high_threshold: float
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
        regime_probability=probabilities,
        dynamics=dynamics,
        frames=tuple(frames),
        medium_threshold=monitor.medium_risk_threshold,
        high_threshold=monitor.high_risk_threshold,
        unit=str(config.get('plotting', {}).get('deformation_unit', '')),
        native_unit=str(
            config.get('plotting', {}).get('native_deformation_rate_unit', '')
        ),
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
                    frame.regime_change_probability,
                    frame.dynamics,
                    frame.risk_level,
                ]
            )
    return path


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
    acceleration_unit = _acceleration_unit(rate_unit)
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    acceleration_axis = axes[1].twinx()
    figure.subplots_adjust(top=0.88, hspace=0.12)
    boundary = datetime.fromisoformat(result.monitoring.start_date)
    calibration_start = datetime.fromisoformat(result.calibration.start_date)
    monitoring_end = datetime.fromisoformat(result.monitoring.end_date)
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
    axes[1].set_title('B. Velocity / acceleration', loc='left')
    axes[2].set_title('C. Regime-change probability / risk', loc='left')
    axes[0].set_ylabel(f'LOS deformation rate [{rate_unit}]')
    axes[1].set_ylabel(f'Velocity [{rate_unit}]', color='#4c78a8')
    acceleration_axis.set_ylabel(
        f'Acceleration [{acceleration_unit}]',
        color='#e45756',
    )
    axes[2].set_ylabel('Probability')
    axes[2].set_ylim(-0.03, 1.03)
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
    observed_line, = axes[0].plot(
        [],
        [],
        color='0.62',
        marker='o',
        markersize=3,
        linewidth=1,
        label='Historical observed rate',
    )
    current_observation, = axes[0].plot(
        [],
        [],
        color='black',
        marker='o',
        markersize=4,
        linestyle='none',
        label='Current observation',
        zorder=5,
    )
    predicted_line, = axes[0].plot(
        [], [], color='#4c78a8', label='Predicted baseline rate'
    )
    velocity_line, = axes[1].plot([], [], color='#4c78a8', label='Velocity')
    acceleration_line, = acceleration_axis.plot(
        [], [], color='#e45756', label='Acceleration'
    )
    probability_line, = axes[2].plot(
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
    axes[2].legend(loc='upper left', fontsize=8)
    velocity_handles, velocity_labels = axes[1].get_legend_handles_labels()
    acceleration_handles, acceleration_labels = acceleration_axis.get_legend_handles_labels()
    axes[1].legend(
        velocity_handles + acceleration_handles,
        velocity_labels + acceleration_labels,
        loc='upper left',
        fontsize=8,
    )

    def update(current: int) -> tuple[Any, ...]:
        visible = np.arange(len(dates)) <= current
        historical = np.arange(len(dates)) < current
        observed_line.set_data(
            dates[historical], result.observed[historical] * scale
        )
        current_observation.set_data(
            [dates[current]], [result.observed[current] * scale]
        )
        monitoring_visible = visible & (
            np.arange(len(dates)) >= result.monitoring.start_index
        )
        predicted_line.set_data(
            dates[monitoring_visible], result.predicted[monitoring_visible] * scale
        )
        velocity_line.set_data(
            dates[monitoring_visible], result.velocity[monitoring_visible] * scale
        )
        acceleration_line.set_data(
            dates[monitoring_visible], result.acceleration[monitoring_visible] * scale
        )
        probability_line.set_data(
            dates[monitoring_visible], result.regime_probability[monitoring_visible]
        )
        for cursor in cursors:
            cursor.set_xdata([dates[current], dates[current]])
        for axis in (axes[0], axes[1], acceleration_axis):
            axis.relim()
            axis.autoscale_view()
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
            velocity_line,
            acceleration_line,
            probability_line,
            *cursors,
            status,
        )

    return figure, update


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

    def resolve(value: object) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (config_path.parent / path).resolve()

    data_config = config['data']
    feature_paths = [
        resolve(data_config[key])
        for key in (
            'features_directory',
            'temporal_features_directory',
            'meteo_features_directory',
        )
        if data_config.get(key)
    ]
    loaded = FeatureLoader(
        list(dict.fromkeys(feature_paths)),
        resolve(dataset_config['mask_path']),
        str(data_config.get('temporal_alignment_method', 'exact')),
    ).load(list(dict.fromkeys([*names, target])), reference_feature=target)
    builder = DatasetBuilder()
    dataset = builder.build(loaded, names, target)
    model_name = str(config['model'])
    model_path = (
        experiment_model_directory(resolve(config['outputs']['root']), config)
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
    LOGGER.info('Wrote monitoring visualization: %s', visual_path)
    LOGGER.info('Wrote monitoring trajectory: %s', csv_path)


if __name__ == '__main__':
    main()
