"""Static slope-stability monitoring dashboard artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.monitoring.temporal_monitoring import TemporalMonitoringResult


def write_slope_stability_dashboard(
    output_path: Path,
    dates: tuple[str, ...],
    monitoring: TemporalMonitoringResult,
    observed_stack: np.ndarray,
    residual_stack: np.ndarray,
    mask: np.ndarray,
    grid_transform: Any,
    grid_width: int,
    grid_height: int,
    unit: str,
    value_scale: float,
    calibration_window: tuple[int, int],
    monitoring_window: tuple[int, int],
    residual_percentile: float,
) -> None:
    """Write a compound dashboard for slope-stability monitoring.

    Args:
        output_path: PNG dashboard destination.
        dates: ISO acquisition dates.
        monitoring: Aggregate residual-monitoring signals.
        observed_stack: Observed temporal raster stack.
        residual_stack: Observation-minus-prediction residual raster stack.
        mask: TSF mask.
        grid_transform: Raster affine transform.
        grid_width: Raster column count.
        grid_height: Raster row count.
        unit: Display unit for deformation quantities.
        value_scale: Conversion from native values to ``unit``.
        calibration_window: Inclusive/exclusive calibration acquisition bounds.
        monitoring_window: Inclusive/exclusive monitoring acquisition bounds.
        residual_percentile: Absolute residual percentile for symmetric map limits.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _validate(
        dates,
        observed_stack,
        residual_stack,
        mask,
        value_scale,
        residual_percentile,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(calibration_window[0], monitoring_window[1])
    figure = plt.figure(figsize=(19, 13), constrained_layout=True)
    grid = GridSpec(4, 3, figure=figure, width_ratios=(1.0, 1.0, 1.15))
    temporal_axes = [figure.add_subplot(grid[index, :2]) for index in range(4)]
    observation_axis = figure.add_subplot(grid[:2, 2])
    residual_axis = figure.add_subplot(grid[2:, 2])

    _plot_predictions(
        temporal_axes[0],
        positions,
        monitoring,
        unit,
        value_scale,
    )
    _plot_anomaly_magnitude(temporal_axes[1], positions, monitoring, unit, value_scale)
    _plot_cusum(temporal_axes[2], positions, monitoring)
    _plot_regime_risk(temporal_axes[3], positions, monitoring)
    for axis in temporal_axes:
        _shade_windows(axis, calibration_window, monitoring_window)
        axis.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
        _format_time_axis(axis, dates, positions)
        axis.grid(alpha=0.25)
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
    _plot_monitoring_residual_mean(
        residual_axis,
        residual_stack,
        monitoring_window,
        mask,
        grid_transform,
        grid_width,
        grid_height,
        unit,
        value_scale,
        residual_percentile,
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
    """Plot mean observations, predictions and predictive uncertainty."""
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
            label='Prediction uncertainty',
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
        label='Threshold',
    )
    axis.set(title='Anomaly magnitude', ylabel=f'Residual rate [{unit}]')


def _plot_cusum(
    axis: Any,
    positions: np.ndarray,
    result: TemporalMonitoringResult,
) -> None:
    """Plot directional CUSUM signals, oscillation and persistent warnings."""
    axis.plot(
        positions,
        result.acceleration_cusum[positions],
        color='red',
        linewidth=1.5,
        label='Acceleration CUSUM',
    )
    axis.plot(
        positions,
        result.deceleration_cusum[positions],
        color='green',
        linewidth=1.5,
        label='Deceleration CUSUM',
    )
    oscillation = positions[result.oscillation[positions]]
    if oscillation.size:
        axis.scatter(
            oscillation,
            np.maximum(
                result.acceleration_cusum[oscillation],
                result.deceleration_cusum[oscillation],
            ),
            color='orange',
            s=20,
            label='Oscillation',
            zorder=3,
        )
    axis.set(
        title='Observed-velocity acceleration CUSUM',
        ylabel='CUSUM statistic',
    )


def _plot_regime_risk(
    axis: Any,
    positions: np.ndarray,
    result: TemporalMonitoringResult,
) -> None:
    """Plot smoothed regime-change risk and configured risk levels."""
    axis.plot(
        positions,
        result.regime_risk[positions],
        color='violet',
        linewidth=1.8,
        label='Smoothed risk',
    )
    axis.axhline(
        result.high_risk_threshold,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label='High risk',
    )
    axis.axhline(
        result.medium_risk_threshold,
        color='orange',
        linestyle='--',
        linewidth=1.2,
        label='Medium risk',
    )
    axis.set(title='Regime change probability', ylabel='Probability', ylim=(0.0, 1.0))


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


def _plot_monitoring_residual_mean(
    axis: Any,
    residual_stack: np.ndarray,
    monitoring_window: tuple[int, int],
    mask: np.ndarray,
    transform: Any,
    width: int,
    height: int,
    unit: str,
    value_scale: float,
    percentile: float,
) -> None:
    """Plot the per-pixel mean residual rate for the monitoring period only."""
    import matplotlib.pyplot as plt

    monitoring_residuals = residual_stack[monitoring_window[0] : monitoring_window[1]]
    finite_count = np.sum(np.isfinite(monitoring_residuals), axis=0)
    values = np.divide(
        np.nansum(monitoring_residuals, axis=0),
        finite_count,
        out=np.full(mask.shape, np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    values = np.where(mask, values * value_scale, np.nan)
    limit = _symmetric_limit(values, percentile)
    extent = _extent(transform, width, height)
    image = axis.imshow(
        values,
        cmap='RdBu_r',
        vmin=-limit,
        vmax=limit,
        extent=extent,
        origin='upper',
    )
    _map_outline(axis, mask, extent)
    colorbar = plt.colorbar(image, ax=axis, shrink=0.72)
    colorbar.set_label(f'Mean residual rate [{unit}]')
    axis.set(
        title='Mean residual during monitoring period',
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
    residual_stack: np.ndarray,
    mask: np.ndarray,
    value_scale: float,
    residual_percentile: float,
) -> None:
    """Validate dashboard raster dimensions and display conversion."""
    if observed.ndim != 3 or residual_stack.shape != observed.shape:
        raise ValueError('Dashboard observation and residual stacks must match in 3D.')
    if observed.shape[0] != len(dates) or observed.shape[1:] != mask.shape:
        raise ValueError(
            'Dashboard dates or mask are incompatible with temporal stacks.'
        )
    if value_scale <= 0:
        raise ValueError('Dashboard value_scale must be positive.')
    if not 0.0 < residual_percentile <= 100.0:
        raise ValueError('Dashboard residual percentile must be in (0, 100].')
