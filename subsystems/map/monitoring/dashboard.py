"""Static slope-stability monitoring dashboard artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.monitoring.level2 import Level2Result
from subsystems.map.monitoring.level3 import Level3Result
from subsystems.map.monitoring.temporal_monitoring import TemporalMonitoringResult


def write_slope_stability_dashboard(
    output_path: Path,
    dates: tuple[str, ...],
    monitoring: TemporalMonitoringResult,
    observed_stack: np.ndarray,
    coherent_acceleration_cusum: np.ndarray,
    level2: Level2Result,
    level3: Level3Result,
    pixel_level2_maximum: np.ndarray,
    mask: np.ndarray,
    grid_transform: Any,
    grid_width: int,
    grid_height: int,
    unit: str,
    value_scale: float,
    calibration_window: tuple[int, int],
    monitoring_window: tuple[int, int],
) -> None:
    """Write a compound dashboard for slope-stability monitoring.

    Args:
        output_path: PNG dashboard destination.
        dates: ISO acquisition dates.
        monitoring: Aggregate residual-monitoring signals.
        observed_stack: Observed temporal raster stack.
        coherent_acceleration_cusum: Maximum CUSUM values for coherence-
            qualified regional acceleration pixels.
        level2: TSF-wide and regional L2 residual CUSUM series.
        level3: L3 regime-change score, thresholds and evidence levels.
        pixel_level2_maximum: Per-pixel maximum L2 instability CUSUM.
        mask: TSF mask.
        grid_transform: Raster affine transform.
        grid_width: Raster column count.
        grid_height: Raster row count.
        unit: Display unit for deformation quantities.
        value_scale: Conversion from native values to ``unit``.
        calibration_window: Inclusive/exclusive calibration acquisition bounds.
        monitoring_window: Inclusive/exclusive monitoring acquisition bounds.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _validate(
        dates,
        observed_stack,
        coherent_acceleration_cusum,
        pixel_level2_maximum,
        mask,
        value_scale,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(calibration_window[0], monitoring_window[1])
    figure = plt.figure(figsize=(19, 16), constrained_layout=True)
    grid = GridSpec(5, 3, figure=figure, width_ratios=(1.0, 1.0, 1.15))
    temporal_axes = [figure.add_subplot(grid[index, :2]) for index in range(5)]
    map_grid = grid[:, 2].subgridspec(3, 1)
    observation_axis = figure.add_subplot(map_grid[0])
    level2_map_axis = figure.add_subplot(map_grid[1])
    residual_axis = figure.add_subplot(map_grid[2])

    _plot_predictions(
        temporal_axes[0],
        positions,
        monitoring,
        unit,
        value_scale,
    )
    _plot_anomaly_magnitude(temporal_axes[1], positions, monitoring, unit, value_scale)
    _plot_level2(temporal_axes[2], positions, level2)
    regional_cusum_axis = _plot_cusum(temporal_axes[3], positions, monitoring)
    _plot_regime_change_score(temporal_axes[4], positions, monitoring, level3)
    for axis in temporal_axes:
        _shade_windows(axis, calibration_window, monitoring_window)
        axis.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
        _format_time_axis(axis, dates, positions)
        axis.grid(alpha=0.25)
        if axis is temporal_axes[3] and regional_cusum_axis is not None:
            left_handles, left_labels = axis.get_legend_handles_labels()
            right_handles, right_labels = (
                regional_cusum_axis.get_legend_handles_labels()
            )
            axis.legend(
                left_handles + right_handles,
                left_labels + right_labels,
                loc='upper left',
                fontsize=8,
                ncols=2,
            )
        else:
            axis.legend(loc='upper left', fontsize=8, ncols=2)

    _plot_latest_observation(
        observation_axis,
        observed_stack,
        dates,
        monitoring_window[1] - 1,
        mask,
        grid_transform,
        grid_width,
        grid_height,
        unit,
        value_scale,
    )
    _plot_level2_map(
        level2_map_axis,
        pixel_level2_maximum,
        level2.parameters.decision_threshold,
        mask,
        grid_transform,
        grid_width,
        grid_height,
    )
    _plot_coherent_acceleration_cusum(
        residual_axis,
        coherent_acceleration_cusum,
        mask,
        grid_transform,
        grid_width,
        grid_height,
    )
    figure.suptitle(
        'Slope stability monitoring dashboard', fontsize=16, fontweight='bold'
    )
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def _plot_predictions(
    axis: Any,
    positions: np.ndarray,
    result: TemporalMonitoringResult,
    unit: str,
    value_scale: float,
) -> None:
    """Plot mean observations, predictions and the ensemble prediction spread."""
    observed = result.observed_mean[positions] * value_scale
    predicted = result.predicted_mean[positions] * value_scale
    axis.scatter(
        positions,
        observed,
        color='black',
        s=12,
        alpha=0.8,
        label='Observed mean LOS velocity',
    )
    axis.scatter(
        positions,
        predicted,
        color='tab:blue',
        s=12,
        alpha=0.75,
        label='Predicted mean LOS velocity',
    )
    if result.uncertainty_mean is not None:
        uncertainty = result.uncertainty_mean[positions] * value_scale
        axis.fill_between(
            positions,
            predicted - uncertainty,
            predicted + uncertainty,
            color='tab:blue',
            alpha=0.2,
            label='Prediction spread (ensemble, not a confidence interval)',
        )
    if result.regional_observed_mean is not None and np.any(
        np.isfinite(result.regional_observed_mean[positions])
    ):
        reg_observed = result.regional_observed_mean[positions] * value_scale
        axis.plot(
            positions,
            reg_observed,
            color='darkorchid',
            linewidth=1.5,
            label='Coherent-region mean LOS velocity',
        )
        axis.scatter(
            positions,
            reg_observed,
            color='darkorchid',
            s=16,
            alpha=0.9,
            zorder=4,
        )
    axis.set(
        title='Mean LOS velocity and baseline prediction',
        ylabel=f'Mean LOS velocity [{unit}]',
    )


def _plot_anomaly_magnitude(
    axis: Any,
    positions: np.ndarray,
    result: TemporalMonitoringResult,
    unit: str,
    value_scale: float,
) -> None:
    """Plot residual anomaly magnitude and its configured decision threshold."""
    axis.plot(
        positions,
        result.anomaly_magnitude[positions] * value_scale,
        color='red',
        linewidth=1.5,
        label='|Mean residual|',
    )
    axis.axhline(
        result.anomaly_threshold * value_scale,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label=f'Anomaly threshold ({result.anomaly_threshold * value_scale:g})',
    )
    axis.set(title='Anomaly magnitude', ylabel=f'Residual rate [{unit}]')


def _plot_level2(
    axis: Any,
    positions: np.ndarray,
    level2: Level2Result,
) -> None:
    """Plot L2 residual CUSUM, its decision threshold and first alarms."""
    threshold = level2.parameters.decision_threshold
    axis.plot(
        positions,
        level2.instability_cusum[positions],
        color='red',
        linewidth=1.5,
        label='TSF-wide L2 instability CUSUM',
    )
    axis.plot(
        positions,
        level2.overforecast_cusum[positions],
        color='0.55',
        linewidth=0.9,
        linestyle='--',
        label='TSF-wide over-forecast CUSUM (diagnostic)',
    )
    if np.any(level2.regional_available[positions]):
        axis.plot(
            positions,
            level2.regional_instability_cusum[positions],
            color='darkorchid',
            linewidth=1.5,
            label='Coherent-region L2 instability CUSUM',
        )
    axis.axhline(
        threshold,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label=f'L2 decision threshold h ({threshold:g})',
    )
    for index, color, label in (
        (level2.first_alarm_index, 'red', 'First TSF-wide L2 alarm'),
        (level2.regional_first_alarm_index, 'darkorchid', 'First regional L2 alarm'),
    ):
        if index is not None and positions[0] <= index <= positions[-1]:
            axis.axvline(index, color=color, linewidth=1.0, alpha=0.7, label=label)
    # Linear up to h keeps the decision level readable when CUSUM runs far past it.
    axis.set_yscale('symlog', linthresh=threshold)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel('CUSUM [sigma units, symlog]')
    axis.set_title(
        'L2: CUSUM of standardized forecast residuals',
        pad=12,
    )


def _plot_cusum(
    axis: Any,
    positions: np.ndarray,
    result: TemporalMonitoringResult,
) -> Any | None:
    """Plot supporting acceleration CUSUM signals and their decision level."""
    axis.plot(
        positions,
        result.acceleration_cusum_instability[positions],
        color='red',
        linewidth=0.9,
        linestyle='--',
        alpha=0.75,
        label='TSF-wide acceleration CUSUM (instability side)',
    )
    axis.plot(
        positions,
        result.acceleration_cusum_recovery[positions],
        color='green',
        linewidth=0.9,
        linestyle='--',
        alpha=0.75,
        label='TSF-wide acceleration CUSUM (recovery side)',
    )
    regional_axis: Any | None = None
    if np.any(result.regional_cusum_available[positions]):
        regional_axis = axis.twinx()
        regional_axis.plot(
            positions,
            result.regional_acceleration_cusum_instability[positions],
            color='darkorchid',
            linewidth=1.5,
            label='Coherent-region acceleration CUSUM (instability side)',
        )
        regional_axis.plot(
            positions,
            result.regional_acceleration_cusum_recovery[positions],
            color='teal',
            linewidth=1.5,
            label='Coherent-region acceleration CUSUM (recovery side)',
        )
        regional_axis.set_ylabel('Coherent-region CUSUM statistic')
        regional_axis.set_ylim(
            _regional_cusum_limits(
                result.regional_acceleration_cusum_instability[positions],
                result.regional_acceleration_cusum_recovery[positions],
            )
        )
        _plot_regional_dynamics_timeline(axis, positions, result)
    axis.axhline(
        result.acceleration_cusum_decision_threshold,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label=(
            'Acceleration decision threshold '
            f'({result.acceleration_cusum_decision_threshold:g})'
        ),
    )
    oscillation = positions[result.oscillation[positions]]
    if oscillation.size:
        axis.scatter(
            oscillation,
            np.maximum(
                result.acceleration_cusum_instability[oscillation],
                result.acceleration_cusum_recovery[oscillation],
            ),
            color='orange',
            s=20,
            label='Oscillation',
            zorder=3,
        )
    axis.set_ylabel('Acceleration CUSUM statistic')
    axis.set_title(
        'Supporting evidence: acceleration CUSUM, TSF-wide and coherent regions',
        pad=12,
    )
    return regional_axis


def _plot_regional_dynamics_timeline(
    axis: Any,
    positions: np.ndarray,
    result: TemporalMonitoringResult,
) -> None:
    """Mark coherent-region acceleration and recovery alarm intervals above plot."""
    styles = {
        'accelerating': ('firebrick', 'Regional acceleration'),
        'decelerating': ('seagreen', 'Regional deceleration / recovery'),
    }
    regional = result.regional_dynamics[positions]
    available = result.regional_cusum_available[positions]
    legend_added: set[str] = set()
    for dynamics, (color, label) in styles.items():
        active = available & (regional == dynamics)
        for start, end in _contiguous_runs(positions, active):
            axis.plot(
                [positions[start], positions[end]],
                [0.90, 0.90],
                color=color,
                linewidth=4.0,
                solid_capstyle='butt',
                transform=axis.get_xaxis_transform(),
                clip_on=True,
                zorder=5,
            )
            if dynamics not in legend_added:
                axis.plot(
                    [],
                    [],
                    color=color,
                    linewidth=4.0,
                    solid_capstyle='butt',
                    label=f'{label} period',
                )
                legend_added.add(dynamics)


def _contiguous_runs(
    positions: np.ndarray,
    active: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Return inclusive index ranges for contiguous true values."""
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _regional_cusum_limits(
    acceleration: np.ndarray,
    deceleration: np.ndarray,
) -> tuple[float, float]:
    """Return a readable min/max scale dedicated to regional CUSUM values."""
    values = np.concatenate((acceleration, deceleration))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lower = min(0.0, float(np.min(values)))
    upper = max(0.0, float(np.max(values)))
    span = upper - lower
    padding = max(0.1, span * 0.08)
    return lower - padding, upper + padding


def _cusum_status(result: TemporalMonitoringResult, index: int) -> str:
    """Return the current TSF-wide persistent CUSUM status for a date."""
    dynamics = str(result.dynamics[index])
    if dynamics == 'accelerating':
        return 'Acceleration alarm'
    if dynamics == 'decelerating':
        return 'Deceleration/recovery signal'
    return 'No CUSUM alarm'


def _cusum_status_color(status: str) -> str:
    """Return a muted status colour without changing the signal line colours."""
    if status == 'Acceleration alarm':
        return 'red'
    if status == 'Deceleration/recovery signal':
        return 'green'
    return '0.6'


def _plot_regime_change_score(
    axis: Any,
    positions: np.ndarray,
    result: TemporalMonitoringResult,
    level3: Level3Result,
) -> None:
    """Plot the heuristic L3 score with watch/alert levels from the evidence rule."""
    for name, color in (('watch', 'orange'), ('alert', 'red')):
        for index in positions[level3.level[positions] == name]:
            axis.axvspan(index - 0.5, index + 0.5, color=color, alpha=0.15, lw=0)
    axis.plot(
        positions,
        result.regime_change_score[positions],
        color='violet',
        linewidth=1.1,
        linestyle='--',
        alpha=0.8,
        label='TSF-wide regime-change score',
    )
    regional_available = result.regional_cusum_available[positions]
    if np.any(regional_available):
        regional_score = np.where(
            regional_available,
            result.regional_regime_change_score[positions],
            np.nan,
        )
        axis.plot(
            positions,
            regional_score,
            color='violet',
            linewidth=2.0,
            label='Coherent-region regime-change score',
        )
    parameters = level3.parameters
    axis.axhline(
        parameters.alert_threshold,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f'Alert score threshold ({parameters.alert_threshold:g})',
    )
    axis.axhline(
        parameters.watch_threshold,
        color='orange',
        linestyle='--',
        linewidth=1.2,
        label=f'Watch score threshold ({parameters.watch_threshold:g})',
    )
    for index, color, label in (
        (level3.first_watch_index, 'orange', 'First L3 watch'),
        (level3.first_alert_index, 'red', 'First L3 alert'),
    ):
        if index is not None:
            axis.axvline(index, color=color, linewidth=1.0, alpha=0.8, label=label)
    axis.set(
        title=(
            'L3: regime-change score (heuristic, not a probability); '
            'shading = L3 watch/alert'
        ),
        ylabel='Score [0–1]',
        ylim=(0.0, 1.0),
    )


def _plot_latest_observation(
    axis: Any,
    observed_stack: np.ndarray,
    dates: tuple[str, ...],
    observation_index: int,
    mask: np.ndarray,
    transform: Any,
    width: int,
    height: int,
    unit: str,
    value_scale: float,
) -> None:
    """Plot the latest InSAR observation in map coordinates."""
    import matplotlib.pyplot as plt

    values = np.where(mask, observed_stack[observation_index] * value_scale, np.nan)
    extent = _extent(transform, width, height)
    limit = _symmetric_limit(values)
    image = axis.imshow(
        values,
        cmap='Greys',
        vmin=-limit,
        vmax=limit,
        extent=extent,
        origin='upper',
    )
    _map_outline(axis, mask, extent)
    colorbar = plt.colorbar(image, ax=axis, shrink=0.72)
    colorbar.set_label(f'LOS deformation rate [{unit}]')
    axis.set(
        title=f'InSAR LOS deformation rate ({dates[observation_index]})',
        xlabel='Easting',
        ylabel='Northing',
    )


def _plot_coherent_acceleration_cusum(
    axis: Any,
    acceleration_cusum_instability: np.ndarray,
    mask: np.ndarray,
    transform: Any,
    width: int,
    height: int,
) -> None:
    """Plot the maximum acceleration CUSUM for coherent monitoring regions."""
    import matplotlib.pyplot as plt

    values = np.where(mask, acceleration_cusum_instability, np.nan)
    finite = values[np.isfinite(values)]
    limit = 1.0 if finite.size == 0 else max(1.0, float(np.max(finite)))
    extent = _extent(transform, width, height)
    image = axis.imshow(
        values,
        cmap='magma',
        vmin=0.0,
        vmax=limit,
        extent=extent,
        origin='upper',
    )
    _map_outline(axis, mask, extent)
    colorbar = plt.colorbar(image, ax=axis, shrink=0.72)
    colorbar.set_label('Acceleration CUSUM statistic')
    axis.set(
        title='Maximum coherence-qualified acceleration CUSUM',
        xlabel='Easting',
        ylabel='Northing',
    )


def _plot_level2_map(
    axis: Any,
    values: np.ndarray,
    threshold: float,
    mask: np.ndarray,
    transform: Any,
    width: int,
    height: int,
) -> None:
    """Plot the per-pixel maximum L2 instability CUSUM with h on the scale."""
    import matplotlib.pyplot as plt

    values = np.where(mask, values, np.nan)
    finite = values[np.isfinite(values)]
    upper = max(threshold * 1.5, float(np.max(finite)) if finite.size else 0.0)
    extent = _extent(transform, width, height)
    image = axis.imshow(
        values, cmap='magma', vmin=0.0, vmax=upper, extent=extent, origin='upper'
    )
    _map_outline(axis, mask, extent)
    colorbar = plt.colorbar(image, ax=axis, shrink=0.72)
    colorbar.ax.axhline(threshold, color='cyan', linewidth=1.5)
    colorbar.set_label('L2 CUSUM [sigma units]; line = h')
    axis.set(
        title='Per-pixel maximum L2 instability CUSUM',
        xlabel='Easting',
        ylabel='Northing',
    )


def _map_outline(
    axis: Any,
    mask: np.ndarray,
    extent: tuple[float, float, float, float],
) -> None:
    """Draw the TSF boundary without adding configured reference points."""
    axis.contour(
        mask.astype(float),
        levels=[0.5],
        colors='black',
        linewidths=1.0,
        extent=extent,
        origin='upper',
    )


def _shade_windows(
    axis: Any,
    calibration_window: tuple[int, int],
    monitoring_window: tuple[int, int],
) -> None:
    """Apply the dashboard's yellow calibration and green monitoring shading."""
    axis.axvspan(
        calibration_window[0] - 0.5,
        calibration_window[1] - 0.5,
        color='#f4d35e',
        alpha=0.2,
        label='Calibration',
        zorder=0,
    )
    axis.axvspan(
        monitoring_window[0] - 0.5,
        monitoring_window[1] - 0.5,
        color='#95d5b2',
        alpha=0.2,
        label='Monitoring',
        zorder=0,
    )


def _format_time_axis(
    axis: Any,
    dates: tuple[str, ...],
    positions: np.ndarray,
) -> None:
    """Add a readable date axis to a dashboard temporal panel."""
    tick_count = min(10, positions.size)
    ticks = np.unique(
        np.linspace(positions[0], positions[-1], tick_count, dtype=int),
    )
    axis.set_xticks(ticks, [dates[index] for index in ticks])
    axis.tick_params(axis='x', rotation=35, labelsize=8)
    axis.set_xlabel('Acquisition date')


def _extent(
    transform: Any, width: int, height: int
) -> tuple[float, float, float, float]:
    """Return an imshow extent from a raster transform and dimensions."""
    left = transform.c
    right = transform.c + transform.a * width
    top = transform.f
    bottom = transform.f + transform.e * height
    return left, right, bottom, top


def _symmetric_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    """Return a robust symmetric colour limit for signed map values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError('Latest observation contains no finite TSF values.')
    if not 0.0 < percentile <= 100.0:
        raise ValueError('Map residual percentile must be in (0, 100].')
    return float(np.percentile(np.abs(finite), percentile)) or 1.0


def _validate(
    dates: tuple[str, ...],
    observed: np.ndarray,
    coherent_acceleration_cusum: np.ndarray,
    pixel_level2_maximum: np.ndarray,
    mask: np.ndarray,
    value_scale: float,
) -> None:
    """Validate dashboard raster dimensions and display conversion."""
    if observed.ndim != 3:
        raise ValueError('Dashboard observations must be a 3D stack.')
    if coherent_acceleration_cusum.shape != mask.shape:
        raise ValueError('Dashboard coherent acceleration CUSUM has an invalid shape.')
    if pixel_level2_maximum.shape != mask.shape:
        raise ValueError('Dashboard per-pixel L2 CUSUM has an invalid shape.')
    if observed.shape[0] != len(dates) or observed.shape[1:] != mask.shape:
        raise ValueError(
            'Dashboard dates or mask are incompatible with temporal stacks.'
        )
    if value_scale <= 0:
        raise ValueError('Dashboard value_scale must be positive.')
