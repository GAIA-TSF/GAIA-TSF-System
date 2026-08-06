"""Artifact and metric helpers shared by learning and inference pipelines."""

from __future__ import annotations

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


def write_diagnostics(
    output_dir: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    dates: tuple[str, ...],
    time_indices: np.ndarray,
) -> None:
    """Write observed/predicted and residual diagnostic PNGs."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = observed - predicted
    figure, axis = plt.subplots()
    axis.scatter(observed, predicted, s=5, alpha=0.35)
    limits = [
        float(np.nanmin((observed, predicted))),
        float(np.nanmax((observed, predicted))),
    ]
    axis.plot(limits, limits, 'k--', linewidth=1)
    axis.set(xlabel='Observed', ylabel='Predicted', title='Observed vs predicted')
    figure.savefig(
        output_dir / 'observed_vs_predicted.png', dpi=150, bbox_inches='tight'
    )
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.hist(residuals[np.isfinite(residuals)], bins=50)
    axis.set(xlabel='Residual', ylabel='Count', title='Residual histogram')
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
    figure, axis = plt.subplots()
    axis.plot(labels, observed_mean, label='observed')
    axis.plot(labels, predicted_mean, label='predicted')
    axis.tick_params(axis='x', rotation=45)
    axis.legend()
    axis.set(title='Mean time-series comparison')
    figure.savefig(
        output_dir / 'timeseries_comparison.png', dpi=150, bbox_inches='tight'
    )
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(
        labels,
        [float(np.mean(residuals[time_indices == index])) for index in unique_times],
    )
    axis.tick_params(axis='x', rotation=45)
    axis.set(ylabel='Mean residual', title='Temporal residual profile')
    figure.savefig(output_dir / 'residual_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(figure)
