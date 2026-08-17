"""Artifact and metric helpers shared by learning and inference pipelines."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import numpy as np


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return deterministic common regression metrics for finite paired samples."""
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(valid):
        raise ValueError('Cannot calculate metrics without finite prediction pairs.')
    residual = observed[valid] - predicted[valid]
    return {
        'rmse': float(np.sqrt(np.mean(np.square(residual)))),
        'mae': float(np.mean(np.abs(residual))),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON artifact, making required directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8'
    )


def _point_display_name(name: str) -> str:
    """Return the user-facing label for a configured observation point."""
    if name == 'deformation_zone':
        return 'Observed deformation zone'
    return name.replace('_', ' ').title()


def write_diagnostics(
    output_dir: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    dates: tuple[str, ...],
    time_indices: np.ndarray,
    unit: str,
    value_scale: float = 1.0,
    uncertainty: np.ndarray | None = None,
    calibration_window: tuple[int, int] | None = None,
    monitoring_window: tuple[int, int] | None = None,
    cumulative_unit: str | None = None,
    fallback_interval_days: float = 1.0,
    pixel_indices: np.ndarray | None = None,
    cumulative_observation_max_points: int | None = None,
) -> None:
    """Write model diagnostics with optional inference uncertainty.

    Args:
        output_dir: Directory that receives the PNG artifacts.
        observed: Observed deformation values for every plotted sample.
        predicted: Corresponding baseline predictions.
        dates: Acquisition dates indexed by ``time_indices``.
        time_indices: Acquisition index for every sample.
        unit: Physical unit appended to value axes.
        value_scale: Factor that converts native values to the plotted unit.
        uncertainty: Optional one-standard-deviation uncertainty per prediction.
        calibration_window: Inclusive/exclusive acquisition bounds for calibration.
        monitoring_window: Inclusive/exclusive acquisition bounds for monitoring.
        cumulative_unit: Output unit for the optional cumulative-rate diagnostic.
        fallback_interval_days: Interval used if acquisition labels are not ISO dates.
        pixel_indices: Optional flattened pixel index for each observed sample.
        cumulative_observation_max_points: Maximum rendered cumulative sample points.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.shape != time_indices.shape:
        raise ValueError('Observed, predicted, and time_indices must have equal shapes.')
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=np.float64)
        if uncertainty.shape != predicted.shape:
            raise ValueError('Uncertainty and predicted values must have equal shapes.')

    if value_scale <= 0:
        raise ValueError('value_scale must be positive.')
    if fallback_interval_days <= 0:
        raise ValueError('fallback_interval_days must be positive.')
    if pixel_indices is not None and pixel_indices.shape != observed.shape:
        raise ValueError('pixel_indices and observed values must have equal shapes.')
    if (
        cumulative_observation_max_points is not None
        and cumulative_observation_max_points < 1
    ):
        raise ValueError('cumulative_observation_max_points must be positive.')
    observed = observed * value_scale
    predicted = predicted * value_scale
    if uncertainty is not None:
        uncertainty = uncertainty * value_scale

    residuals = observed - predicted
    unit_label = f' [{unit}]' if unit else ''
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        observed,
        predicted,
        s=12,
        alpha=0.35,
        label='Predictions',
    )
    limits = [
        float(np.nanmin((observed, predicted))),
        float(np.nanmax((observed, predicted))),
    ]
    axis.plot(limits, limits, 'k--', linewidth=1, label='Ideal prediction')
    axis.tick_params(axis='x', rotation=90)
    axis.legend()
    axis.grid(alpha=0.25)
    axis.set(
        xlabel=f'Observed{unit_label}',
        ylabel=f'Predicted{unit_label}',
        title='Observed vs predicted',
    )
    figure.savefig(
        output_dir / 'observed_vs_predicted.png', dpi=150, bbox_inches='tight'
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    finite_residuals = residuals[np.isfinite(residuals)]
    axis.hist(
        finite_residuals,
        bins=50,
        color='tab:blue',
        alpha=0.8,
        edgecolor='white',
        label='Residuals',
    )
    axis.axvline(
        float(np.mean(finite_residuals)),
        color='tab:red',
        linestyle='--',
        linewidth=1.5,
        label='Mean residual',
    )
    axis.legend()
    axis.grid(axis='y', alpha=0.25)
    axis.set(
        xlabel=f'Residual{unit_label}',
        ylabel='Count',
        title='Residual histogram',
    )
    figure.savefig(output_dir / 'residual_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(figure)

    unique_times = np.unique(time_indices)
    observed_mean = [
        float(np.mean(observed[time_indices == index])) for index in unique_times
    ]
    predicted_mean = [
        float(np.mean(predicted[time_indices == index])) for index in unique_times
    ]
    labels = [dates[index] for index in unique_times]
    positions = np.arange(unique_times.size)
    observed_groups = [
        observed[(time_indices == index) & np.isfinite(observed)]
        for index in unique_times
    ]
    figure, axis = plt.subplots(figsize=(14, 6))
    boxplot = axis.boxplot(
        observed_groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )
    for box in boxplot['boxes']:
        box.set(facecolor='tab:gray', alpha=0.25)
    boxplot['boxes'][0].set_label('Observed distribution')
    for median in boxplot['medians']:
        median.set(color='tab:gray')
    axis.plot(
        positions,
        observed_mean,
        color='black',
        linewidth=2,
        label='Observed mean',
    )
    axis.plot(
        positions,
        predicted_mean,
        color='tab:blue',
        linewidth=2,
        label='Predicted mean',
    )
    _shade_temporal_windows(
        axis,
        positions,
        unique_times,
        calibration_window,
        monitoring_window,
    )
    axis.set_xticks(positions, labels, rotation=45, ha='right')
    axis.legend()
    axis.grid(alpha=0.25)
    axis.set(
        xlabel='Acquisition date',
        ylabel=f'Deformation{unit_label}',
        title='Mean time-series comparison',
    )
    figure.savefig(
        output_dir / 'timeseries_comparison.png', dpi=150, bbox_inches='tight'
    )
    plt.close(figure)

    residual_mean = [
        float(np.mean(residuals[time_indices == index])) for index in unique_times
    ]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.scatter(
        np.searchsorted(unique_times, time_indices),
        residuals,
        s=10,
        alpha=0.2,
        color='tab:purple',
        label='Residual samples',
    )
    axis.plot(
        positions,
        residual_mean,
        color='tab:purple',
        linewidth=2,
        label='Mean residual',
    )
    axis.axhline(0.0, color='black', linestyle='--', linewidth=1)
    _shade_temporal_windows(
        axis,
        positions,
        unique_times,
        calibration_window,
        monitoring_window,
    )
    axis.set_xticks(positions, labels, rotation=45, ha='right')
    axis.legend()
    axis.grid(alpha=0.25)
    axis.set(
        xlabel='Acquisition date',
        ylabel=f'Residual{unit_label}',
        title='Temporal residual profile',
    )
    figure.savefig(output_dir / 'residual_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(figure)

    if uncertainty is not None:
        uncertainty_mean = [
            float(np.mean(uncertainty[time_indices == index]))
            for index in unique_times
        ]
        figure, axis = plt.subplots(figsize=(16, 4))
        predicted_mean_array = np.asarray(predicted_mean)
        uncertainty_mean_array = np.asarray(uncertainty_mean)
        axis.fill_between(
            positions,
            predicted_mean_array - uncertainty_mean_array,
            predicted_mean_array + uncertainty_mean_array,
            color='tab:blue',
            alpha=0.2,
            label='Prediction uncertainty',
        )
        axis.scatter(
            np.searchsorted(unique_times, time_indices),
            observed,
            color='grey',
            s=1,
            alpha=0.05,
            label='Observed samples',
            zorder=2,
        )
        axis.scatter(
            positions,
            observed_mean,
            color='black',
            s=20,
            label='Observed mean',
            zorder=4,
        )
        axis.plot(
            positions,
            predicted_mean,
            color='tab:blue',
            linewidth=2,
            label='Predicted mean',
        )
        _shade_temporal_windows(
            axis,
            positions,
            unique_times,
            calibration_window,
            monitoring_window,
        )
        axis.set_xticks(positions, labels, rotation=45, ha='right')
        axis.set(
            xlabel='Acquisition date',
            ylabel=f'Deformation{unit_label}',
            title='Prediction with uncertainty',
        )
        axis.legend(loc='best')
        axis.grid(alpha=0.25)
        figure.savefig(
            output_dir / 'prediction_with_uncertainty.png',
            dpi=150,
            bbox_inches='tight',
        )
        plt.close(figure)

    if cumulative_unit is not None:
        _write_cumulative_displacement_diagnostic(
            output_dir=output_dir,
            positions=positions,
            labels=labels,
            date_lookup=dates,
            observed_rate=np.asarray(observed_mean),
            predicted_rate=np.asarray(predicted_mean),
            predicted_uncertainty=(
                None if uncertainty is None else np.asarray(uncertainty_mean)
            ),
            time_values=unique_times,
            calibration_window=calibration_window,
            monitoring_window=monitoring_window,
            cumulative_unit=cumulative_unit,
            fallback_interval_days=fallback_interval_days,
            observed_samples=observed,
            observed_time_indices=time_indices,
            observed_positions=np.searchsorted(unique_times, time_indices),
            observed_pixel_indices=pixel_indices,
            cumulative_observation_max_points=cumulative_observation_max_points,
        )


def _write_cumulative_displacement_diagnostic(
    output_dir: Path,
    positions: np.ndarray,
    time_values: np.ndarray,
    labels: list[str],
    date_lookup: tuple[str, ...],
    observed_rate: np.ndarray,
    predicted_rate: np.ndarray,
    predicted_uncertainty: np.ndarray | None,
    calibration_window: tuple[int, int] | None,
    monitoring_window: tuple[int, int] | None,
    cumulative_unit: str,
    fallback_interval_days: float,
    observed_samples: np.ndarray,
    observed_time_indices: np.ndarray,
    observed_positions: np.ndarray,
    observed_pixel_indices: np.ndarray | None,
    cumulative_observation_max_points: int | None,
) -> None:
    """Integrate mean rate observations and predictions into cumulative displacement."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    intervals = _acquisition_intervals_days(labels, fallback_interval_days)
    observed_cumulative = _integrate_rate(observed_rate, intervals)
    predicted_cumulative = _integrate_rate(predicted_rate, intervals)
    figure, axis = plt.subplots(figsize=(20, 6))
    _shade_temporal_windows(
        axis,
        positions,
        time_values,
        calibration_window,
        monitoring_window,
    )
    if predicted_uncertainty is not None:
        cumulative_uncertainty = _integrate_uncertainty(
            predicted_uncertainty,
            intervals,
        )
        axis.fill_between(
            positions,
            predicted_cumulative - cumulative_uncertainty,
            predicted_cumulative + cumulative_uncertainty,
            color='tab:blue',
            alpha=0.2,
            label='Prediction uncertainty',
        )
    if observed_pixel_indices is not None:
        observation_positions, observation_cumulative = _cumulative_observation_points(
            dates=date_lookup,
            observed_samples=observed_samples,
            time_indices=observed_time_indices,
            plot_positions=observed_positions,
            pixel_indices=observed_pixel_indices,
            fallback_interval_days=fallback_interval_days,
            maximum_points=cumulative_observation_max_points,
        )
        axis.scatter(
            observation_positions,
            observation_cumulative,
            color='black',
            s=4,
            alpha=0.035,
            rasterized=True,
            label='Observed pixel trajectories',
            zorder=1,
        )
    axis.plot(
        positions,
        observed_cumulative,
        color='black',
        linewidth=2,
        label='Observed cumulative displacement',
    )
    axis.plot(
        positions,
        predicted_cumulative,
        color='tab:blue',
        linewidth=2,
        label='Predicted cumulative displacement',
    )
    tick_positions = _time_tick_positions(len(labels))
    axis.set_xticks(tick_positions, [labels[index] for index in tick_positions])
    axis.tick_params(axis='x', rotation=30)
    axis.set(
        xlabel='Acquisition date',
        ylabel=f'Cumulative displacement [{cumulative_unit}]',
        title='Cumulative displacement comparison',
    )
    axis.legend(loc='best')
    axis.grid(alpha=0.25)
    figure.savefig(
        output_dir / 'cumulative_displacement_comparison.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(figure)


def _cumulative_observation_points(
    dates: list[str],
    observed_samples: np.ndarray,
    time_indices: np.ndarray,
    plot_positions: np.ndarray,
    pixel_indices: np.ndarray,
    fallback_interval_days: float,
    maximum_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate rendered cumulative trajectories for individual observed pixels."""
    positions: list[np.ndarray] = []
    cumulative_values: list[np.ndarray] = []
    for pixel_index in np.unique(pixel_indices):
        include = pixel_indices == pixel_index
        pixel_times = time_indices[include]
        pixel_positions = plot_positions[include]
        pixel_values = observed_samples[include]
        order = np.argsort(pixel_times)
        pixel_times = pixel_times[order]
        pixel_positions = pixel_positions[order]
        pixel_values = pixel_values[order]
        intervals = _acquisition_intervals_days(
            [dates[index] for index in pixel_times],
            fallback_interval_days,
        )
        positions.append(pixel_positions)
        cumulative_values.append(_integrate_rate(pixel_values, intervals))
    if not positions:
        return np.array([], dtype=int), np.array([], dtype=np.float64)
    all_positions = np.concatenate(positions)
    all_values = np.concatenate(cumulative_values)
    if maximum_points is not None and all_values.size > maximum_points:
        selection = np.linspace(
            0,
            all_values.size - 1,
            num=maximum_points,
            dtype=int,
        )
        return all_positions[selection], all_values[selection]
    return all_positions, all_values


def write_observation_point_timeseries(
    output_dir: Path,
    observed: np.ndarray,
    dates: tuple[str, ...],
    time_indices: np.ndarray,
    pixel_indices: np.ndarray,
    grid_transform: Any,
    grid_width: int,
    grid_height: int,
    points: dict[str, dict[str, object]],
    unit: str,
    value_scale: float = 1.0,
    cumulative_unit: str | None = None,
    fallback_interval_days: float = 1.0,
    window_size: int = 3,
    calibration_window: tuple[int, int] | None = None,
    monitoring_window: tuple[int, int] | None = None,
) -> None:
    """Write observation time series for named map coordinates.

    Args:
        output_dir: Directory that receives the PNG artifact.
        observed: Native observed target values for valid dataset samples.
        dates: Acquisition labels indexed by ``time_indices``.
        time_indices: Acquisition index for every dataset sample.
        pixel_indices: Flattened grid pixel index for every dataset sample.
        grid_transform: Raster affine transform used to locate coordinates.
        grid_width: Number of raster columns.
        grid_height: Number of raster rows.
        points: Named point configuration containing ``coordinates: [x, y]``.
        unit: Physical unit appended to the y-axis.
        value_scale: Factor converting native values to the displayed unit.
        cumulative_unit: Output unit for the optional cumulative point diagnostic.
        fallback_interval_days: Interval used if acquisition labels are not ISO dates.
        window_size: Odd pixel-window width and height centred on each point.
        calibration_window: Inclusive/exclusive acquisition bounds for calibration.
        monitoring_window: Inclusive/exclusive acquisition bounds for monitoring.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not points:
        return
    if value_scale <= 0:
        raise ValueError('value_scale must be positive.')
    if fallback_interval_days <= 0:
        raise ValueError('fallback_interval_days must be positive.')
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError('window_size must be a positive odd integer.')
    if len(observed) != len(time_indices) or len(observed) != len(pixel_indices):
        raise ValueError('Point time-series sample arrays must have equal lengths.')

    figure, axis = plt.subplots(figsize=(16, 4.5))
    point_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    point_records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for series_index, (name, point_config) in enumerate(points.items()):
        pixel_indices_in_window = _point_window_pixel_indices(
            name,
            point_config,
            grid_transform,
            grid_width,
            grid_height,
            window_size,
        )
        include = np.isin(pixel_indices, pixel_indices_in_window)
        if not np.any(include):
            raise ValueError(
                f"Configured point '{name}' has no valid MAP observation samples "
                f'in its {window_size}x{window_size} pixel window.',
            )
        record_times = time_indices[include]
        record_observed = np.asarray(observed[include], dtype=np.float64) * value_scale
        record_pixels = pixel_indices[include]
        point_records[name] = (record_pixels, record_times, record_observed)
        point_times = np.unique(record_times)
        point_observed = np.asarray(
            [np.mean(record_observed[record_times == index]) for index in point_times],
        )
        point_series[name] = (point_times, point_observed)
        display_name = _point_display_name(name)
        color = 'tab:orange' if name == 'deformation_zone' else f'C{series_index}'
        axis.scatter(
            record_times,
            record_observed,
            color=color,
            s=12,
            alpha=0.14,
            label=f'{display_name} 3x3 records',
        )
        axis.plot(
            point_times,
            point_observed,
            color=color,
            linewidth=2,
            label=f'{display_name} 3x3 mean',
        )
        axis.scatter(
            point_times,
            point_observed,
            color=color,
            s=22,
            alpha=0.8,
        )

    tick_positions = _time_tick_positions(len(dates))
    axis.set_xticks(tick_positions, [dates[index] for index in tick_positions])
    _shade_temporal_windows(
        axis,
        np.arange(len(dates)),
        np.arange(len(dates)),
        calibration_window,
        monitoring_window,
    )
    axis.tick_params(axis='x', rotation=45)
    axis.set(
        xlabel='Acquisition date',
        ylabel=f'Observed deformation [{unit}]' if unit else 'Observed deformation',
        title='Observed deformation at configured TSF points (3x3 pixels)',
    )
    axis.legend()
    axis.grid(alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / 'point_observation_timeseries.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(figure)

    if cumulative_unit is not None:
        _write_point_cumulative_displacement_diagnostic(
            output_dir=output_dir,
            dates=dates,
            point_series=point_series,
            point_records=point_records,
            cumulative_unit=cumulative_unit,
            fallback_interval_days=fallback_interval_days,
            window_size=window_size,
            calibration_window=calibration_window,
            monitoring_window=monitoring_window,
        )


def write_latest_residual_map(
    output_dir: Path,
    residual_stack: np.ndarray,
    dates: tuple[str, ...],
    mask: np.ndarray,
    grid_transform: Any,
    grid_width: int,
    grid_height: int,
    points: dict[str, dict[str, object]],
    unit: str,
    value_scale: float,
    colormap: str,
    percentile: float,
    aggregation_window: int = 1,
    fallback_interval_days: float = 1.0,
    cumulative_unit: str | None = None,
    output_filename: str = 'residual_latest.png',
    title_override: str | None = None,
) -> None:
    """Write a spatial map for the latest residual acquisition.

    Args:
        output_dir: Directory that receives the PNG artifact.
        residual_stack: Residual raster stack shaped ``(time, rows, columns)``.
        dates: Acquisition dates corresponding to the residual stack.
        mask: TSF mask used to outline the monitored facility.
        grid_transform: Raster affine transform for map-coordinate plotting.
        grid_width: Number of raster columns.
        grid_height: Number of raster rows.
        points: Named point configuration containing ``coordinates: [x, y]``.
        unit: Physical unit shown on the colour bar.
        value_scale: Factor converting native residuals to the displayed unit.
        colormap: Matplotlib diverging colour map name.
        percentile: Absolute-residual percentile used for symmetric colour limits.
        aggregation_window: Latest residual acquisitions integrated into the map.
        fallback_interval_days: Interval used if acquisition labels are not ISO dates.
        cumulative_unit: Unit used when integrating multiple rate observations.
        output_filename: Name of the generated PNG artifact.
        title_override: Optional title replacing the default map title.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if residual_stack.ndim != 3 or residual_stack.shape[1:] != mask.shape:
        raise ValueError('Residual stack and TSF mask have incompatible dimensions.')
    if len(dates) != residual_stack.shape[0]:
        raise ValueError('Residual stack and acquisition dates have incompatible lengths.')
    if value_scale <= 0:
        raise ValueError('value_scale must be positive.')
    if not 0 < percentile <= 100:
        raise ValueError('Residual colour percentile must be in the interval (0, 100].')
    if aggregation_window < 1:
        raise ValueError('Residual aggregation_window must be positive.')
    if fallback_interval_days <= 0:
        raise ValueError('fallback_interval_days must be positive.')

    window = min(aggregation_window, residual_stack.shape[0])
    if window == 1:
        values = np.asarray(residual_stack[-1], dtype=np.float64) * value_scale
        map_unit = unit
        acquisition_label = dates[-1]
        title = f'Latest TSF residual map ({acquisition_label})'
    else:
        recent_residuals = np.asarray(residual_stack[-window:], dtype=np.float64)
        recent_dates = list(dates[-window:])
        intervals = _acquisition_intervals_days(
            recent_dates,
            fallback_interval_days,
        )
        increments = 0.5 * (
            recent_residuals[1:] + recent_residuals[:-1]
        ) * intervals[1:, np.newaxis, np.newaxis]
        valid_increment = np.any(np.isfinite(increments), axis=0)
        values = np.where(
            valid_increment,
            np.nansum(increments, axis=0) * value_scale,
            np.nan,
        )
        map_unit = cumulative_unit or unit
        acquisition_label = f'{recent_dates[0]} to {recent_dates[-1]}'
        title = (
            f'Cumulative TSF residual ({window} latest acquisitions, '
            f'{acquisition_label})'
        )
    if title_override is not None:
        title = title_override
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError('Latest residual raster contains no finite values.')
    limit = float(np.percentile(np.abs(finite_values), percentile))
    if limit == 0:
        limit = float(np.max(np.abs(finite_values))) or 1.0

    left = grid_transform.c
    right = grid_transform.c + grid_transform.a * grid_width
    top = grid_transform.f
    bottom = grid_transform.f + grid_transform.e * grid_height
    extent = (left, right, bottom, top)
    figure, axis = plt.subplots(figsize=(11, 8))
    image = axis.imshow(
        values,
        cmap=colormap,
        vmin=-limit,
        vmax=limit,
        extent=extent,
        origin='upper',
    )
    axis.contour(
        mask.astype(float),
        levels=[0.5],
        colors='black',
        linewidths=1.2,
        extent=extent,
        origin='upper',
    )
    for name, point_config in points.items():
        coordinates = point_config.get('coordinates')
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 2
            or not all(isinstance(value, (int, float)) for value in coordinates)
        ):
            raise ValueError(
                f"Configured point '{name}' requires coordinates: [x, y].",
            )
        x_coordinate, y_coordinate = coordinates
        axis.scatter(
            x_coordinate,
            y_coordinate,
            s=55,
            edgecolor='black',
            linewidth=0.8,
            label=_point_display_name(name),
        )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    quantity_label = 'Cumulative residual displacement' if window > 1 else 'Residual'
    colorbar.set_label(
        f'{quantity_label} [{map_unit}]' if map_unit else quantity_label,
    )
    axis.set(
        xlabel='Easting',
        ylabel='Northing',
        title=title,
    )
    if points:
        axis.legend(loc='best')
    figure.savefig(
        output_dir / output_filename,
        dpi=180,
        bbox_inches='tight',
    )
    plt.close(figure)


def write_mean_residual_map(
    output_dir: Path,
    residual_stack: np.ndarray,
    dates: tuple[str, ...],
    mask: np.ndarray,
    grid_transform: Any,
    grid_width: int,
    grid_height: int,
    points: dict[str, dict[str, object]],
    unit: str,
    value_scale: float,
    colormap: str,
    percentile: float,
) -> None:
    """Write the per-pixel mean residual map across all acquisitions."""
    if residual_stack.ndim != 3:
        raise ValueError('Residual stack must have shape (time, rows, columns).')
    finite_count = np.sum(np.isfinite(residual_stack), axis=0)
    mean_residual = np.divide(
        np.nansum(residual_stack, axis=0),
        finite_count,
        out=np.full(residual_stack.shape[1:], np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    write_latest_residual_map(
        output_dir=output_dir,
        residual_stack=mean_residual[np.newaxis, :, :],
        dates=('all acquisitions',),
        mask=mask,
        grid_transform=grid_transform,
        grid_width=grid_width,
        grid_height=grid_height,
        points=points,
        unit=unit,
        value_scale=value_scale,
        colormap=colormap,
        percentile=percentile,
        output_filename='residual_mean.png',
        title_override=f'Mean TSF residual ({len(dates)} acquisitions)',
    )


def write_persistent_residual_map(
    output_dir: Path,
    persistent_anomalies: np.ndarray,
    mask: np.ndarray,
    grid_transform: Any,
    grid_width: int,
    grid_height: int,
    colormap: str,
    persistence_start_time_index: int = 0,
    persistence_end_time_index: int | None = None,
    persistence_fraction_display_max: float = 1.0,
) -> None:
    """Write a map of post-calibration persistent-anomaly fractions.

    Args:
        output_dir: Directory that receives the PNG artifact.
        persistent_anomalies: Boolean anomaly stack shaped ``(time, rows, columns)``.
        mask: TSF mask used to outline the monitored facility.
        grid_transform: Raster affine transform for map-coordinate plotting.
        grid_width: Number of raster columns.
        grid_height: Number of raster rows.
        colormap: Matplotlib sequential colour map name.
        persistence_start_time_index: First acquisition eligible for persistence.
        persistence_end_time_index: Exclusive final acquisition eligible for
            persistence.
        persistence_fraction_display_max: Fixed upper colour-bar limit for
            persistence fractions.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if persistent_anomalies.ndim != 3 or persistent_anomalies.shape[1:] != mask.shape:
        raise ValueError('Persistent anomalies and TSF mask have incompatible dimensions.')
    time_count = persistent_anomalies.shape[0]
    if persistence_end_time_index is None:
        persistence_end_time_index = time_count
    if not 0 <= persistence_start_time_index < persistence_end_time_index <= time_count:
        raise ValueError(
            'Persistence analysis window must reference at least one acquisition.',
        )
    if not 0.0 < persistence_fraction_display_max <= 1.0:
        raise ValueError('Persistence fraction display maximum must be in (0, 1].')
    eligible = persistent_anomalies[
        persistence_start_time_index:persistence_end_time_index
    ]
    persistence_fraction = np.mean(eligible.astype(np.float64), axis=0)
    persistence_fraction = np.where(mask, persistence_fraction, np.nan)
    maximum_fraction = float(np.nanmax(persistence_fraction))
    left = grid_transform.c
    right = grid_transform.c + grid_transform.a * grid_width
    top = grid_transform.f
    bottom = grid_transform.f + grid_transform.e * grid_height
    extent = (left, right, bottom, top)
    figure, axis = plt.subplots(figsize=(11, 8))
    image = axis.imshow(
        persistence_fraction,
        cmap=colormap,
        vmin=0.0,
        vmax=persistence_fraction_display_max,
        extent=extent,
        origin='upper',
    )
    axis.contour(
        mask.astype(float),
        levels=[0.5],
        colors='black',
        linewidths=1.2,
        extent=extent,
        origin='upper',
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label(
        f'Persistent anomaly fraction [0–{persistence_fraction_display_max:.3f}]',
    )
    axis.set(
        xlabel='Easting',
        ylabel='Northing',
        title=(
            'Post-calibration persistent residual anomaly fraction '
            f'({eligible.shape[0]} acquisitions; max {maximum_fraction:.3f})'
        ),
    )
    figure.savefig(
        output_dir / 'residual_persistent.png',
        dpi=180,
        bbox_inches='tight',
    )
    plt.close(figure)

def _write_point_cumulative_displacement_diagnostic(
    output_dir: Path,
    dates: tuple[str, ...],
    point_series: dict[str, tuple[np.ndarray, np.ndarray]],
    point_records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    cumulative_unit: str,
    fallback_interval_days: float,
    window_size: int,
    calibration_window: tuple[int, int] | None,
    monitoring_window: tuple[int, int] | None,
) -> None:
    """Write relative cumulative displacement time series for configured points."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(16, 4.5))
    for series_index, (name, (time_indices, observed_rate)) in enumerate(
        point_series.items(),
    ):
        color = f'C{series_index}'
        record_pixels, record_times, record_observed = point_records[name]
        for pixel_index in np.unique(record_pixels):
            include = record_pixels == pixel_index
            pixel_times = record_times[include]
            pixel_rates = record_observed[include]
            order = np.argsort(pixel_times)
            pixel_times = pixel_times[order]
            pixel_rates = pixel_rates[order]
            pixel_dates = [dates[index] for index in pixel_times]
            pixel_intervals = _acquisition_intervals_days(
                pixel_dates,
                fallback_interval_days,
            )
            axis.scatter(
                pixel_times,
                _integrate_rate(pixel_rates, pixel_intervals),
                color=color,
                s=12,
                alpha=0.14,
                label='_nolegend_',
            )
        point_dates = [dates[index] for index in time_indices]
        intervals = _acquisition_intervals_days(point_dates, fallback_interval_days)
        cumulative = _integrate_rate(observed_rate, intervals)
        axis.plot(
            time_indices,
            cumulative,
            color=color,
            linewidth=2,
            label=f'{_point_display_name(name)} {window_size}x{window_size} mean',
        )
        axis.scatter(time_indices, cumulative, s=18, alpha=0.65)

    tick_positions = _time_tick_positions(len(dates))
    axis.set_xticks(tick_positions, [dates[index] for index in tick_positions])
    _shade_temporal_windows(
        axis,
        np.arange(len(dates)),
        np.arange(len(dates)),
        calibration_window,
        monitoring_window,
    )
    axis.tick_params(axis='x', rotation=45)
    axis.set(
        xlabel='Acquisition date',
        ylabel=f'Cumulative displacement [{cumulative_unit}]',
        title=(
            'Cumulative displacement at configured TSF points '
            f'({window_size}x{window_size} pixels)'
        ),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(
        output_dir / 'point_cumulative_displacement_timeseries.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(figure)


def _point_window_pixel_indices(
    name: str,
    point_config: dict[str, object],
    grid_transform: Any,
    grid_width: int,
    grid_height: int,
    window_size: int,
) -> np.ndarray:
    """Return flattened pixel indices from a centred configured coordinate window."""
    coordinates = point_config.get('coordinates')
    if (
        not isinstance(coordinates, list)
        or len(coordinates) != 2
        or not all(isinstance(value, (int, float)) for value in coordinates)
    ):
        raise ValueError(
            f"Configured point '{name}' requires coordinates: [x, y].",
        )
    x_coordinate, y_coordinate = (float(value) for value in coordinates)
    column_float, row_float = ~grid_transform * (x_coordinate, y_coordinate)
    row, column = int(np.floor(row_float)), int(np.floor(column_float))
    if not (0 <= row < grid_height and 0 <= column < grid_width):
        raise ValueError(
            f"Configured point '{name}' is outside the monitored raster: "
            f'({x_coordinate}, {y_coordinate}).',
        )
    radius = window_size // 2
    rows, columns = np.mgrid[
        max(0, row - radius) : min(grid_height, row + radius + 1),
        max(0, column - radius) : min(grid_width, column + radius + 1),
    ]
    return (rows * grid_width + columns).ravel()


def _time_tick_positions(date_count: int, maximum_ticks: int = 12) -> np.ndarray:
    """Choose evenly distributed acquisition ticks for readable long time series."""
    if date_count <= maximum_ticks:
        return np.arange(date_count)
    return np.unique(np.linspace(0, date_count - 1, maximum_ticks, dtype=int))


def _shade_temporal_windows(
    axis: Any,
    positions: np.ndarray,
    time_values: np.ndarray,
    calibration_window: tuple[int, int] | None,
    monitoring_window: tuple[int, int] | None,
) -> None:
    """Shade configured calibration and monitoring intervals on a temporal axis."""
    for window, color, label in (
        (calibration_window, '#f4d35e', 'Calibration period'),
        (monitoring_window, '#9ecae1', 'Monitoring period'),
    ):
        if window is None:
            continue
        start_index, end_index = window
        start_position = int(np.searchsorted(time_values, start_index, side='left'))
        end_position = int(np.searchsorted(time_values, end_index, side='left'))
        if start_position >= len(positions) or end_position <= 0:
            continue
        left = positions[max(start_position, 0)] - 0.5
        right = positions[min(end_position, len(positions)) - 1] + 0.5
        axis.axvspan(left, right, color=color, alpha=0.2, label=label, zorder=0)


def _acquisition_intervals_days(
    labels: list[str],
    fallback_interval_days: float,
) -> np.ndarray:
    """Return elapsed days between acquisition labels, using a safe configured fallback."""
    intervals = np.zeros(len(labels), dtype=np.float64)
    try:
        acquisition_dates = [date.fromisoformat(label) for label in labels]
    except ValueError:
        intervals[1:] = fallback_interval_days
        return intervals
    intervals[1:] = [
        (current - previous).days
        for previous, current in zip(acquisition_dates, acquisition_dates[1:])
    ]
    return intervals


def _integrate_rate(rate: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Integrate rate values with the trapezoidal rule over acquisition intervals."""
    cumulative = np.zeros(rate.size, dtype=np.float64)
    for index in range(1, rate.size):
        cumulative[index] = cumulative[index - 1] + (
            0.5 * (rate[index - 1] + rate[index]) * intervals[index]
        )
    return cumulative


def _integrate_uncertainty(
    uncertainty: np.ndarray,
    intervals: np.ndarray,
) -> np.ndarray:
    """Propagate independent rate uncertainty through trapezoidal integration."""
    variance = np.zeros(uncertainty.size, dtype=np.float64)
    for index in range(1, uncertainty.size):
        weight = 0.5 * intervals[index]
        variance[index] = variance[index - 1] + weight**2 * (
            uncertainty[index - 1] ** 2 + uncertainty[index] ** 2
        )
    return np.sqrt(variance)
