#!/usr/bin/env python3
"""Animate LOS observations, normal behaviour and Level-1 residual anomalies.

z = (observed - expected) / sigma. Flag |z| > threshold within monitoring.
Sigma is supplied explicitly or estimated from calibration residuals (ddof=1).
A Level-1 anomaly is a single unusual observation, not a warning.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

# Support direct execution from any working directory.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np

from subsystems.map.scripts.visualisation.plot_monitoring_observations import load_data


def window_mask(payload: dict, dates: list[date], name: str) -> np.ndarray:
    """Use calendar bounds because export indices refer to the original stack."""
    window = payload.get('temporal_windows', {}).get(name, {})
    start = window.get('configured_start_date')
    end = window.get('configured_end_date')
    if start or end:
        lower = date.fromisoformat(start) if start else None
        upper = date.fromisoformat(end) if end else None
        return np.array([
            (lower is None or day >= lower) and
            (upper is None or (day <= upper if name == 'monitoring' else day < upper))
            for day in dates
        ])
    return np.array([row.get('phase') == name for row in payload['series']])


def level1_scores(payload: dict, dates: list[date], values: dict,
                  sigma: float | None = None, threshold: float = 3.0):
    """Return z, monitored anomaly flags, sigma and calibration sample count.

    Residuals are not demeaned: this follows the standardized-residual formula
    on the slide. Undefined residuals remain NaN. Sigma uses only the configured
    calibration window; overlapping full-period calibration is retrospective.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('The z-score threshold must be finite and positive.')
    residual = values['observed_mean_los_velocity'] - values['predicted_baseline_velocity']
    calibration = window_mask(payload, dates, 'calibration') & np.isfinite(residual)
    count = int(calibration.sum())
    if sigma is None:
        if count < 2:
            raise ValueError(
                'Need at least two finite calibration residuals to estimate sigma. '
                'This export may omit calibration predictions. Supply --sigma '
                'from stable-area validation residuals, in the JSON velocity units.'
            )
        sigma = float(np.std(residual[calibration], ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Sigma must be finite and positive; zero residual variance cannot define z-scores.')
    z = residual / sigma
    anomalies = window_mask(payload, dates, 'monitoring') & np.isfinite(z) & (np.abs(z) > threshold)
    return z, anomalies, float(sigma), count


def plot_level1(input_path: Path, output: Path, *, sigma: float | None = None,
                threshold: float = 3.0, fps: float = 8, dpi: int = 120,
                marker_size: float = 6, title: str | None = None) -> Path:
    """Create a static PNG/PDF/SVG or a progressively revealed MP4/GIF."""
    payload, dates, values = load_data(input_path)
    z, anomalies, sigma_used, count = level1_scores(payload, dates, values, sigma, threshold)
    print(f'Sigma: {sigma_used:g}; source: ' +
          ('explicit' if sigma is not None else f'{count} calibration residuals') +
          f'; Level-1 anomalies: {anomalies.sum()}')
    observed = values['observed_mean_los_velocity']
    predicted = values['predicted_baseline_velocity']
    numbers = mdates.date2num(dates)
    unit = payload.get('units', {}).get('deformation_rate', 'native units')
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout='constrained')
    heading = title or 'InSAR LOS monitoring — Level 1'
    heading_artist = fig.suptitle(heading)
    histories, currents, anomaly_points = [], [], []
    try:
        for index, axis in enumerate(axes):
            series = observed if index < 2 else z
            history, = axis.plot([], [], 'o-', color='#999999', markersize=marker_size,
                                 linewidth=1.1, label='Historical observations' if index < 2 else 'Standardized residual')
            current, = axis.plot([], [], 'o', color='black', markersize=marker_size + 2,
                                 label='Current acquisition', zorder=5)
            histories.append(history); currents.append(current)
            red, = axis.plot([], [], 'o', color='#dc2828', markersize=marker_size + 1,
                             label='Level-1 anomaly', zorder=6)
            anomaly_points.append(red)
            axis.grid(alpha=0.2)
            axis.axhline(0, color='gray', linewidth=0.6, alpha=0.5)
            finite = series[np.isfinite(series)]
            low, high = (float(finite.min()), float(finite.max())) if finite.size else (-1., 1.)
            if index == 1:
                finite_pred = predicted[np.isfinite(predicted)]
                if finite_pred.size:
                    low, high = min(low, finite_pred.min()), max(high, finite_pred.max())
            if index == 2:
                low, high = min(low, -threshold), max(high, threshold)
            padding = max((high-low)*0.12, abs(high)*0.05, 0.001)
            axis.set_ylim(low-padding, high+padding)
            axis.set_ylabel(f'LOS velocity [{unit}]' if index < 2 else 'Residual z-score')
        baseline, = axes[1].plot([], [], color='#2675b5', linewidth=1.8,
                                 label='Predicted normal behaviour')
        for sign in [-1, 1]:
            axes[2].axhline(sign * threshold, color='#dc2828', linestyle='--',
                              linewidth=1, label=f'±{threshold:g} threshold' if sign == 1 else None)
        axes[0].set_title('A. InSAR LOS observations', loc='left')
        axes[1].set_title('B. Comparison with normal behaviour', loc='left')
        axes[2].set_title(
            f'C. Level-1 anomaly detection: z = (observed − expected) / σ; σ = {sigma_used:.4g} {unit}', loc='left')
        start = payload.get('temporal_windows', {}).get('monitoring', {}).get('configured_start_date')
        cursors = []
        for axis in axes:
            if start:
                axis.axvline(date.fromisoformat(start), color='#18aaa0', linestyle='--', linewidth=1)
            cursors.append(axis.axvline(numbers[0], color='black', alpha=0.3, linestyle=':'))
            axis.set_xlim(numbers[0]-1, numbers[-1]+1)
        # The top panel remains an unclassified observation view.
        anomaly_points[0].set_label('_nolegend_')
        for axis in axes:
            axis.legend(loc='upper left', fontsize=8, ncol=3)
        locator = mdates.AutoDateLocator()
        axes[2].xaxis.set_major_locator(locator)
        axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[2].set_xlabel('Acquisition date — a Level-1 anomaly alone is not a warning')

        def update(index):
            visible = slice(0, index + 1)
            for panel, (history, current, red, cursor) in enumerate(
                    zip(histories, currents, anomaly_points, cursors)):
                series = observed if panel < 2 else z
                history.set_data(numbers[visible], series[visible])
                current.set_data([numbers[index]], [series[index]])
                current.set_color('#dc2828' if panel > 0 and anomalies[index] else 'black')
                if panel > 0:
                    flagged = anomalies[visible]
                    red.set_data(numbers[visible][flagged], series[visible][flagged])
                cursor.set_xdata([numbers[index], numbers[index]])
            baseline.set_data(numbers[visible], predicted[visible])
            heading_artist.set_text(f'{heading}  |  {dates[index].isoformat()}')

        output.parent.mkdir(parents=True, exist_ok=True)
        update(len(dates)-1)
        fig.canvas.draw()
        fig.set_layout_engine(None)
        if output.suffix.lower() in {'.mp4', '.gif'}:
            animation = FuncAnimation(fig, update, frames=len(dates), interval=1000/fps,
                                      blit=False, repeat=False)
            writer = (FFMpegWriter(fps=fps, extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
                      if output.suffix.lower() == '.mp4' else PillowWriter(fps=fps))
            animation.save(output, writer=writer, dpi=dpi)
        else:
            fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Exported tsf_monitoring.json')
    parser.add_argument('--output', type=Path, help='MP4 (default), GIF, PNG, PDF or SVG')
    parser.add_argument('--sigma', type=float, help='Normal residual standard deviation in JSON velocity units')
    parser.add_argument('--z-threshold', type=float, default=3)
    parser.add_argument('--fps', type=float, default=8)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--marker-size', type=float, default=6)
    parser.add_argument('--title')
    args = parser.parse_args()
    if not all(np.isfinite(value) and value > 0 for value in [args.fps, args.dpi, args.marker_size]):
        parser.error('--fps, --dpi and --marker-size must be finite and positive')
    output = args.output or args.input.with_name(args.input.stem + '_level1.mp4')
    if output.suffix.lower() not in {'.mp4', '.gif', '.png', '.pdf', '.svg'}:
        parser.error('Unsupported output extension')
    try:
        print(plot_level1(args.input, output, sigma=args.sigma, threshold=args.z_threshold,
                          fps=args.fps, dpi=args.dpi, marker_size=args.marker_size, title=args.title))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
