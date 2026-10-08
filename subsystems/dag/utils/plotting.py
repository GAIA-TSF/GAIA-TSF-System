"""Reusable non-interactive plotting helpers for DAG diagnostic artifacts."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_figure(style: str | None) -> None:
    if style:
        plt.style.use(style)


def _padded_limits(
    values: np.ndarray,
    fraction: float,
    lower_bound: float | None = None,
) -> tuple[float, float]:
    """Return finite data limits with a fraction of their range on each side."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError('Cannot calculate plot limits without finite values')
    minimum, maximum = float(finite.min()), float(finite.max())
    span = maximum - minimum
    if span == 0:
        span = max(abs(minimum), 1.0) * 0.01
    padding = span * fraction
    lower = minimum - padding
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    return lower, maximum + padding


def save_temporal_mean_std_plot(
    dates: tuple[date, ...],
    means: np.ndarray,
    stds: np.ndarray,
    output_path: Path,
    dpi: int,
    style: str | None = None,
    mean_ylim: tuple[float, float] | list[float] | None = None,
    std_ylim: tuple[float, float] | list[float] | None = None,
    axis_padding_fraction: float = 0.30,
    percentile_values: np.ndarray | None = None,
    percentile: float | None = None,
) -> None:
    """Save temporal mean and standard deviation plot with optional fixed axes."""
    _prepare_figure(style)
    frame = pd.DataFrame(
        {
            'date': pd.to_datetime([value.isoformat() for value in dates]),
            'mean': means,
            'std': stds,
        },
    )

    fig, ax_mean = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax_std = ax_mean.twinx()
    ax_mean.plot(frame['date'], frame['mean'], marker='o', label='Mean LOS')
    mean_limit_values = means
    if percentile_values is not None:
        if len(percentile_values) != len(dates):
            raise ValueError('percentile_values length must match dates')
        percentile_label = (
            f'Lower-tail LOS ({percentile:g}th percentile)'
            if percentile is not None
            else 'Lower-tail LOS percentile'
        )
        ax_mean.plot(
            frame['date'],
            percentile_values,
            color='tab:red',
            marker='^',
            label=percentile_label,
        )
        mean_limit_values = np.concatenate((means, percentile_values))
    ax_std.plot(
        frame['date'],
        frame['std'],
        color='tab:orange',
        marker='s',
        label='Temporal std',
    )
    ax_mean.set_xlabel('Acquisition date')
    ax_mean.set_ylabel('Mean LOS displacement')
    ax_std.set_ylabel('Temporal standard deviation')
    if not 0 <= axis_padding_fraction <= 1:
        raise ValueError('axis_padding_fraction must be between 0 and 1')
    if mean_ylim is not None:
        if len(mean_ylim) != 2 or mean_ylim[0] >= mean_ylim[1]:
            raise ValueError('mean_ylim must contain two increasing values')
        ax_mean.set_ylim(float(mean_ylim[0]), float(mean_ylim[1]))
    else:
        ax_mean.set_ylim(_padded_limits(mean_limit_values, axis_padding_fraction))
    if std_ylim is not None:
        if len(std_ylim) != 2 or std_ylim[0] >= std_ylim[1]:
            raise ValueError('std_ylim must contain two increasing values')
        ax_std.set_ylim(float(std_ylim[0]), float(std_ylim[1]))
    else:
        ax_std.set_ylim(_padded_limits(stds, axis_padding_fraction, lower_bound=0))

    # Combined legend
    lines1, labels1 = ax_mean.get_legend_handles_labels()
    lines2, labels2 = ax_std.get_legend_handles_labels()
    ax_mean.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax_mean.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax_mean.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_histogram(
    values: np.ndarray,
    bins: int,
    output_path: Path,
    dpi: int,
    style: str | None = None,
) -> None:
    """Save a global LOS histogram."""
    _prepare_figure(style)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(values, bins=bins, color='steelblue', edgecolor='black', alpha=0.85)
    ax.set_xlabel('LOS displacement')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', alpha=0.3)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_boxplot(
    data: np.ndarray,
    dates: tuple[date, ...],
    output_path: Path,
    dpi: int,
    style: str | None = None,
) -> None:
    """Save LOS distribution boxplots over time."""
    _prepare_figure(style)
    distributions = [layer[np.isfinite(layer)] for layer in data]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(1, len(distributions) + 1)
    ax.boxplot(distributions, positions=positions, showfliers=False)
    tick_count = min(12, len(dates))
    tick_indices = np.unique(
        np.linspace(0, len(dates) - 1, tick_count, dtype=int)
    )
    ax.set_xticks(positions[tick_indices])
    ax.set_xticklabels([dates[index].isoformat() for index in tick_indices])
    ax.set_xlabel('Acquisition date')
    ax.set_ylabel('LOS displacement')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_cumulative_displacement_plot(
    cumulative_data: np.ndarray,
    dates: tuple[date, ...],
    output_path: Path,
    dpi: int,
    style: str | None = None,
    lower_percentile: float = 5.0,
    displacement_unit: str = '',
) -> None:
    """Plot TSF cumulative LOS displacement and its deforming lower tail."""
    if cumulative_data.ndim != 3 or cumulative_data.shape[0] != len(dates):
        raise ValueError(
            'Cumulative displacement stack must match the acquisition dates.'
        )
    if not 0 <= lower_percentile <= 100:
        raise ValueError('lower_percentile must be between 0 and 100.')
    _prepare_figure(style)
    temporal_median = np.nanmedian(cumulative_data, axis=(1, 2))
    lower_quartile = np.nanpercentile(cumulative_data, 25.0, axis=(1, 2))
    upper_quartile = np.nanpercentile(cumulative_data, 75.0, axis=(1, 2))
    lower_tail = np.nanpercentile(
        cumulative_data,
        lower_percentile,
        axis=(1, 2),
    )
    date_values = pd.to_datetime([value.isoformat() for value in dates])

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.fill_between(
        date_values,
        lower_quartile,
        upper_quartile,
        color='0.75',
        alpha=0.55,
        label='Spatial interquartile range',
    )
    axis.plot(
        date_values,
        temporal_median,
        color='black',
        linewidth=1.8,
        label='Spatial median',
    )
    axis.plot(
        date_values,
        lower_tail,
        color='tab:red',
        linewidth=1.8,
        label=f'Negative tail ({lower_percentile:g}th percentile)',
    )
    axis.axhline(0.0, color='0.45', linewidth=0.8)
    axis.set_title('Cumulative LOS displacement')
    axis.set_xlabel('Acquisition date')
    unit_suffix = f' [{displacement_unit}]' if displacement_unit else ''
    axis.set_ylabel(f'Cumulative LOS displacement{unit_suffix}')
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis.grid(True, alpha=0.3)
    axis.legend(loc='best')
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_feature_summary_plot(
    values: np.ndarray,
    output_path: Path,
    feature_name: str,
    *,
    unit: str = '',
    dpi: int = 200,
    histogram_bins: int = 50,
    robust_percentile: float = 98.0,
    sequential_cmap: str = 'viridis',
    diverging_cmap: str = 'RdBu_r',
    extent: tuple[float, float, float, float] | None = None,
    style: str | None = None,
) -> None:
    """Save one spatial-map and distribution summary for a 2D feature."""
    if values.ndim != 2:
        raise ValueError('Feature summary plots require a two-dimensional array.')
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f'Feature {feature_name} contains no finite values.')
    if histogram_bins < 1:
        raise ValueError('Feature summary histogram_bins must be positive.')
    if not 50 < robust_percentile <= 100:
        raise ValueError('Feature summary robust_percentile must be in (50, 100].')

    lower_percentile = 100.0 - robust_percentile
    lower, upper = np.percentile(
        finite,
        [lower_percentile, robust_percentile],
    )
    diverging = lower < 0.0 < upper
    if diverging:
        limit = max(abs(float(lower)), abs(float(upper)))
        vmin, vmax = -limit, limit
        cmap = diverging_cmap
    else:
        vmin, vmax = float(lower), float(upper)
        if vmin == vmax:
            padding = max(abs(vmin), 1.0) * 0.01
            vmin, vmax = vmin - padding, vmax + padding
        cmap = sequential_cmap

    _prepare_figure(style)
    figure, (map_axis, histogram_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
        gridspec_kw={'width_ratios': (1.15, 1.0)},
    )
    image = map_axis.imshow(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin='upper',
    )
    title = feature_name.replace('_', ' ').title()
    unit_suffix = f' [{unit}]' if unit else ''
    map_axis.set_title(f'{title}: spatial distribution')
    map_axis.set_xlabel('Easting' if extent is not None else 'Column')
    map_axis.set_ylabel('Northing' if extent is not None else 'Row')
    colorbar = figure.colorbar(image, ax=map_axis, fraction=0.046, pad=0.04)
    colorbar.set_label(f'{title}{unit_suffix}')

    histogram_axis.hist(
        finite,
        bins=histogram_bins,
        color='steelblue',
        edgecolor='black',
        alpha=0.82,
    )
    mean = float(np.mean(finite))
    median = float(np.median(finite))
    histogram_axis.axvline(
        mean,
        color='tab:red',
        linewidth=1.5,
        label=f'Mean: {mean:.4g}',
    )
    histogram_axis.axvline(
        median,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'Median: {median:.4g}',
    )
    histogram_axis.set_title(f'{title}: value distribution')
    histogram_axis.set_xlabel(f'{title}{unit_suffix}')
    histogram_axis.set_ylabel('Pixel count')
    histogram_axis.grid(True, axis='y', alpha=0.3)
    histogram_axis.legend(loc='best')
    figure.suptitle(f'{title} feature summary', fontweight='bold')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_temporal_feature_summary_plot(
    values: np.ndarray,
    dates: tuple[date, ...],
    output_path: Path,
    feature_name: str,
    *,
    unit: str = '',
    dpi: int = 200,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    style: str | None = None,
) -> None:
    """Plot spatial distribution statistics through time for one feature stack."""
    if values.ndim != 3 or values.shape[0] != len(dates):
        raise ValueError('Temporal feature stack must match the acquisition dates.')
    if not 0 <= lower_percentile < 50:
        raise ValueError('Temporal lower_percentile must be in [0, 50).')
    if not 50 < upper_percentile <= 100:
        raise ValueError('Temporal upper_percentile must be in (50, 100].')

    median = np.full(len(dates), np.nan, dtype=np.float64)
    lower_quartile = np.full(len(dates), np.nan, dtype=np.float64)
    upper_quartile = np.full(len(dates), np.nan, dtype=np.float64)
    lower_tail = np.full(len(dates), np.nan, dtype=np.float64)
    upper_tail = np.full(len(dates), np.nan, dtype=np.float64)
    for index, layer in enumerate(values):
        finite = layer[np.isfinite(layer)]
        if finite.size == 0:
            continue
        (
            lower_tail[index],
            lower_quartile[index],
            median[index],
            upper_quartile[index],
            upper_tail[index],
        ) = np.percentile(
            finite,
            [lower_percentile, 25.0, 50.0, 75.0, upper_percentile],
        )
    if not np.any(np.isfinite(median)):
        raise ValueError(f'Temporal feature {feature_name} has no finite values.')

    _prepare_figure(style)
    date_values = pd.to_datetime([value.isoformat() for value in dates])
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.fill_between(
        date_values,
        lower_quartile,
        upper_quartile,
        color='0.75',
        alpha=0.55,
        label='Spatial interquartile range',
    )
    axis.plot(
        date_values,
        lower_tail,
        color='tab:red',
        linewidth=1.5,
        label=f'Lower tail ({lower_percentile:g}th percentile)',
    )
    axis.plot(
        date_values,
        upper_tail,
        color='tab:blue',
        linewidth=1.5,
        label=f'Upper tail ({upper_percentile:g}th percentile)',
    )
    axis.plot(
        date_values,
        median,
        color='black',
        linewidth=1.8,
        label='Spatial median',
    )
    axis.axhline(0.0, color='0.45', linewidth=0.8)
    title = feature_name.replace('_', ' ').title()
    unit_suffix = f' [{unit}]' if unit else ''
    axis.set_title(f'{title}: temporal behaviour')
    axis.set_xlabel('Acquisition date')
    axis.set_ylabel(f'{title}{unit_suffix}')
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis.grid(True, alpha=0.3)
    axis.legend(loc='best', ncols=2)
    figure.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_heatmap(
    values: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    dpi: int,
    cmap: str,
    style: str | None = None,
    color_limits: tuple[float, float] | list[float] | None = None,
    color_padding_fraction: float = 0.30,
    nonnegative: bool = False,
) -> None:
    """Save a raster heatmap with configured or data-derived color limits."""
    _prepare_figure(style)
    if not 0 <= color_padding_fraction <= 1:
        raise ValueError('color_padding_fraction must be between 0 and 1')
    if color_limits is not None:
        if len(color_limits) != 2 or color_limits[0] >= color_limits[1]:
            raise ValueError('color_limits must contain two increasing values')
        vmin, vmax = float(color_limits[0]), float(color_limits[1])
    else:
        vmin, vmax = _padded_limits(
            values,
            color_padding_fraction,
            lower_bound=0 if nonnegative else None,
        )
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
