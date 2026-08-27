"""Reusable non-interactive plotting helpers for DAG diagnostic artifacts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import os

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


def save_temporal_mean_std_plot(
    dates: tuple[date, ...],
    means: np.ndarray,
    stds: np.ndarray,
    output_path: Path,
    dpi: int,
    style: str | None = None,
) -> None:
    """Save temporal mean and standard deviation plot."""
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
) -> None:
    """Save a raster heatmap."""
    _prepare_figure(style)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
