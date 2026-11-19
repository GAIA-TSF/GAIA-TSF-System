"""Model-diagnostic plots and metrics; written only when plotting is enabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE and MAE for finite pairs (used by permutation importance)."""
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(valid):
        raise ValueError('Cannot calculate metrics without finite prediction pairs.')
    residual = observed[valid] - predicted[valid]
    return {
        'rmse': float(np.sqrt(np.mean(np.square(residual)))),
        'mae': float(np.mean(np.abs(residual))),
    }


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def write_learning_curve(
    output_dir: Path, training: list[float], validation: list[float] | None = None
) -> None:
    """Write epoch training and validation losses when the model records them."""
    if not training:
        return
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(np.arange(1, len(training) + 1), training, label='Training loss')
    if validation:
        axis.plot(np.arange(1, len(validation) + 1), validation, label='Validation loss')
    axis.set(xlabel='Epoch', ylabel='MSE (normalized change)', title='Learning curve')
    axis.grid(alpha=0.25)
    axis.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / 'learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close(figure)


def write_observed_vs_predicted(
    output_dir: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    units: str,
    label: str,
    max_points: int = 50000,
) -> None:
    """Scatter observed against predicted values, one panel per lead."""
    plt = _pyplot()
    horizon = observed.shape[1]
    figure, axes = plt.subplots(1, horizon, figsize=(4 * horizon, 4), squeeze=False)
    rng = np.random.default_rng(0)
    for lead, axis in enumerate(axes[0]):
        valid = np.flatnonzero(np.isfinite(observed[:, lead]) & np.isfinite(predicted[:, lead]))
        if valid.size > max_points:
            valid = rng.choice(valid, max_points, replace=False)
        x, y = predicted[valid, lead], observed[valid, lead]
        axis.scatter(x, y, s=2, alpha=0.3)
        if valid.size:
            low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
            axis.plot([low, high], [low, high], color='black', linewidth=0.8)
        axis.set(
            xlabel=f'Predicted [{units}]',
            ylabel=f'Observed [{units}]',
            title=f'Lead {lead + 1} (n={valid.size})',
        )
    figure.suptitle(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / 'observed_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close(figure)


def write_error_by_lead(
    output_dir: Path, metrics: list[dict[str, Any]], units: str, label: str
) -> None:
    """Plot model and constant-baseline MAE/RMSE against lead."""
    plt = _pyplot()
    leads = [row['lead'] for row in metrics]
    figure, axis = plt.subplots(figsize=(6, 4))
    for key, style in (('mae', '-o'), ('rmse', '-s')):
        axis.plot(leads, [row[key] for row in metrics], style, label=f'model {key.upper()}')
        axis.plot(
            leads,
            [row[f'constant_{key}'] for row in metrics],
            style,
            alpha=0.5,
            label=f'constant {key.upper()}',
        )
    axis.set(xlabel='Lead [acquisitions]', ylabel=f'Error [{units}]', title=label, xticks=leads)
    axis.grid(alpha=0.25)
    axis.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / 'error_by_lead.png', dpi=150, bbox_inches='tight')
    plt.close(figure)
