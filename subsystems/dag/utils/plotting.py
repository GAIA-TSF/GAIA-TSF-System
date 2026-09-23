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
    # labels = [value.isoformat() for value in dates]
    # print(labels)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.boxplot(distributions, showfliers=False)  # tick_labels=labels,
    ax.set_xlabel('Acquisition date')
    ax.set_ylabel('LOS displacement')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


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
