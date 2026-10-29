#!/usr/bin/env python3
"""Plot LOS observations above observations with baseline and monitoring results.

Input is the tsf_monitoring.json exported alongside the MAP monitoring animation.
Values are already in the JSON's display units; no rescaling is applied.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.path import Path as PlotPath
from matplotlib.transforms import Affine2D
import numpy as np


def load_data(path: Path) -> tuple[dict, list[date], dict[str, np.ndarray]]:
    """Read chronological graph data, preserving nulls as gaps."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    records = payload.get('series')
    if not isinstance(records, list) or not records:
        raise ValueError('Input JSON must contain a nonempty series list.')
    dates = [date.fromisoformat(row['date']) for row in records]
    if any(right <= left for left, right in zip(dates, dates[1:])):
        raise ValueError('Acquisition dates must be unique and chronological.')
    keys = ['observed_mean_los_velocity', 'predicted_baseline_velocity',
            'prediction_uncertainty', 'regime_change_score']
    arrays = {key: np.asarray([row.get(key) for row in records], dtype=float)
              for key in keys}
    if not np.any(np.isfinite(arrays['observed_mean_los_velocity'])):
        raise ValueError('No finite LOS observations are available.')
    return payload, dates, arrays


def shade_periods(ax, dates, selected, color, alpha=0.15):
    """Shade contiguous selected acquisitions using midpoint cell boundaries."""
    numbers = mdates.date2num(dates)
    edges = np.r_[numbers[0], (numbers[:-1] + numbers[1:]) / 2,
                  numbers[-1] if len(numbers) > 1 else numbers[0] + 1]
    padded = np.r_[False, np.asarray(selected, dtype=bool), False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    stops = np.flatnonzero(np.diff(padded.astype(int)) == -1)
    for start, stop in zip(starts, stops):
        ax.axvspan(edges[start], edges[stop], color=color, alpha=alpha, linewidth=0)


def plot_observations(input_path: Path, output: Path, *, title: str | None = None,
                      dpi: int = 180, marker_size: float = 6,
                      fps: float = 8) -> Path:
    """Write a static figure or an acquisition-by-acquisition MP4/GIF replay."""
    payload, dates, values = load_data(input_path)
    observed = values['observed_mean_los_velocity']
    predicted = values['predicted_baseline_velocity']
    unit = payload.get('units', {}).get('deformation_rate', 'native units')
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             sharey=True, layout='constrained')
    try:
        histories, currents = [], []
        for ax in axes:
            history, = ax.plot(dates, observed, color='#999999', marker='o',
                               markersize=marker_size, linewidth=1.2,
                               label='Historical LOS observations', zorder=3)
            current, = ax.plot([], [], color='black', marker='o', linestyle='none',
                               markersize=marker_size + 2,
                               label='Current LOS observation', zorder=6)
            histories.append(history)
            currents.append(current)
            ax.set_ylabel(f'LOS velocity [{unit}]')
            ax.grid(alpha=0.2)
            ax.axhline(0, color='gray', linewidth=0.6, alpha=0.5)
        axes[0].set_title('A. InSAR LOS observations', loc='left')
        axes[1].set_title('B. Observations, normal behaviour and monitoring', loc='left')
        ax = axes[1]
        if np.any(np.isfinite(predicted)):
            ax.plot(dates, predicted, color='#2675b5', linewidth=1.8,
                    label='Predicted normal behaviour', zorder=4)
            uncertainty = values['prediction_uncertainty']
            if np.any(np.isfinite(uncertainty) & (uncertainty >= 0)):
                uncertainty = np.where(uncertainty >= 0, uncertainty, np.nan)
                ax.fill_between(dates, predicted - uncertainty, predicted + uncertainty,
                                color='#2675b5', alpha=0.15,
                                label='Baseline ± exported uncertainty')
        # Use calendar bounds, not phase labels or global array indices: windows
        # can overlap, and exported records can start at a nonzero source index.
        monitoring = payload.get('temporal_windows', {}).get('monitoring', {})
        start = monitoring.get('configured_start_date')
        end = monitoring.get('configured_end_date')
        active = np.array([(not start or day >= date.fromisoformat(start))
                           and (not end or day <= date.fromisoformat(end))
                           for day in dates])
        if not start and not end:
            active = np.array([row.get('phase') == 'monitoring'
                               for row in payload['series']])
        score = values['regime_change_score']
        thresholds = payload.get('thresholds', {})
        medium = float(thresholds.get('regime_medium_threshold', 0.5))
        high = float(thresholds.get('regime_high_threshold', 0.8))
        if not 0 <= medium <= high <= 1:
            raise ValueError('Regime thresholds must satisfy 0 <= medium <= high <= 1.')
        shade_periods(ax, dates, active & (score >= medium) & (score < high), '#e5a524')
        shade_periods(ax, dates, active & (score >= high), '#d43d3d')
        handles, labels = ax.get_legend_handles_labels()
        for flag, color, label in [
            ('regional_acceleration_period', '#b52227', 'Regional acceleration'),
            ('regional_deceleration_recovery_period', '#228555', 'Regional recovery'),
        ]:
            selected = active & np.array([bool(row.get(flag, False))
                                          for row in payload['series']])
            if selected.any():
                # Mark intervals in axes coordinates, independent of velocity scale.
                edges = mdates.date2num(dates)
                for index in np.flatnonzero(selected):
                    left = (edges[index-1] + edges[index])/2 if index else edges[index]
                    right = (edges[index] + edges[index+1])/2 if index+1 < len(edges) else edges[index]
                    ax.plot([left, right], [0.96, 0.96], transform=ax.get_xaxis_transform(),
                            color=color, linewidth=5, solid_capstyle='butt')
                handles.append(Patch(color=color)); labels.append(label)
        handles.extend([Patch(facecolor='#e5a524', alpha=0.3),
                        Patch(facecolor='#d43d3d', alpha=0.3)])
        labels.extend([f'Regime score {medium:g}–{high:g}', f'Regime score ≥ {high:g}'])
        ax.legend(handles, labels, loc='best', fontsize=8, ncol=2)
        axes[0].legend(loc='best', fontsize=9)
        for axis in axes:
            if start:
                axis.axvline(date.fromisoformat(start), color='#97723d',
                             linestyle='--', linewidth=0.9)
            axis.margins(x=0.01)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel('Acquisition date')
        heading = title or 'InSAR LOS observations and TSF monitoring'
        heading_artist = fig.suptitle(heading, fontsize=15)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() in {'.mp4', '.gif'}:
            from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

            # Keep the full-period axes fixed, revealing only data up to the
            # current acquisition. Clipping also hides future monitoring bands.
            left = mdates.date2num(dates[0])
            clips = []
            for axis in axes:
                clip = Affine2D().scale(0, 1).translate(left, 0)
                clips.append(clip)
                for artist in [*axis.lines, *axis.collections, *axis.patches]:
                    if artist not in currents:
                        artist.set_clip_path(PlotPath.unit_rectangle(),
                                             clip + axis.get_xaxis_transform())
            cursors = [axis.axvline(dates[0], color='black', alpha=0.4,
                                   linestyle=':', linewidth=1) for axis in axes]

            def update(index):
                for history, current, clip, cursor in zip(histories, currents, clips, cursors):
                    history.set_data(dates[:index + 1], observed[:index + 1])
                    # NaN observations leave the current marker absent rather
                    # than falsely moving it to a previous valid observation.
                    current.set_data([dates[index]], [observed[index]])
                    clip.clear().scale(mdates.date2num(dates[index]) - left, 1).translate(left, 0)
                    # Current markers keep the normal axes clip, so markers at
                    # the reveal boundary are drawn as full circles.
                    cursor.set_xdata([dates[index], dates[index]])
                heading_artist.set_text(f'{heading}  |  {dates[index].isoformat()}')

            fig.canvas.draw()
            fig.set_layout_engine(None)
            animation = FuncAnimation(fig, update, frames=len(dates),
                                      interval=1000 / fps, blit=False, repeat=False)
            writer = FFMpegWriter(fps=fps) if output.suffix.lower() == '.mp4' else PillowWriter(fps=fps)
            animation.save(output, writer=writer, dpi=dpi)
        else:
            for current in currents:
                current.set_data([dates[-1]], [observed[-1]])
            fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Exported tsf_monitoring.json')
    parser.add_argument('--output', type=Path, help='Output PNG, PDF, SVG, MP4 or GIF; defaults beside JSON')
    parser.add_argument('--title', help='Optional figure title')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--marker-size', type=float, default=6, help='Historical marker diameter in points (default: 6)')
    parser.add_argument('--fps', type=float, default=8, help='Animation acquisitions per second (default: 8)')
    args = parser.parse_args()
    if args.marker_size <= 0 or args.fps <= 0:
        parser.error('--marker-size and --fps must be positive')
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    output = args.output or args.input.with_name(args.input.stem + '_observations.png')
    if output.suffix.lower() not in {'.png', '.pdf', '.svg', '.mp4', '.gif'}:
        parser.error('--output must end in .png, .pdf, .svg, .mp4 or .gif')
    print(plot_observations(args.input, output, title=args.title, dpi=args.dpi,
                            marker_size=args.marker_size, fps=args.fps))


if __name__ == '__main__':
    main()
