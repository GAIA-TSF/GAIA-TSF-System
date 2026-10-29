"""Animate level-2 persistence monitoring on one shared temporal plot.

Usage:
    python3 subsystems/map/scripts/animate_level2_cusum.py --output /tmp/level2_normal.gif
    python3 subsystems/map/scripts/animate_level2_cusum.py \
        --case single-anomaly --output /tmp/level2_single_anomaly.mp4
    python3 subsystems/map/scripts/animate_level2_cusum.py \
        --case persistent --output /tmp/level2_persistent.mp4

Cases A, B, and C are approximately transcribed from the supplied images.
The raster does not provide exact acquisition dates, baseline parameters, or
numeric samples: dates are evenly spaced from January 2015 to March 2018 and
LOS values are visual estimates in mm. The expected band is approximated by
a baseline declining from 0 to -5 mm, +/- 10 mm. CUSUM is calculated here, not
digitized from the schematic green curves. All cases use the same parameters.
In case B the source green curve crosses its threshold while stating no
Level-2 warning; this demo retains its defined CUSUM rule and parameters rather
than copying that inconsistent schematic score. The isolated -20 mm observation
is highlighted as Level 1, but the computed CUSUM stays below the Level-2 threshold.

For standardized residual z=(LOS-baseline)/sigma, compute both directions:
    C+ = max(0, previous C+ + z - k)
    C- = max(0, previous C- - z - k)
The displayed statistic is max(C+, C-); a warning occurs at C >= h.
The illustrative sigma=5 mm, k=0.5, h=7.5 are demonstration parameters,
not calibrated operational monitoring settings.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


# Approximate marker heights read left-to-right from the user-provided image.
NORMAL_LOS_MM = (
    0, 0, -3, -1, 0, 2, -2, -1, -4, -4, -2, 0, -4, -3, -7,
    -3, -3, -2, -6, -3, -4, -4, -3, -4, -4, -2, -8, -7, -6,
    -2, -7, -5, -7, -7, -8, -8, -8, -8, -7, -7, -8, -6, -3,
    -4, -4, -6, -6, -7, -6, -7, -9, -9, -6, -5, -3, -6, -8, -8,
)

# Case B: isolated excursion around October 2015, followed by normal variability.
SINGLE_ANOMALY_LOS_MM = (
    0, 0, -3, -1, 0, 2, -2, -4, -3, -1, -2, -1, -4, -10, -20,
    -3, -3, -3, -5, -3, -4, -3, -3, -3, -4, -1, -8, -6, -5,
    -2, -8, -6, -5, -7, -7, -8, -7, -8, -8, -7, -8, -7, -5,
    -6, -5, -4, -7, -7, -6, -7, -7, -7, -8, -9, -7, -5, -9, -8,
)

# Case C: sustained downward departure beginning around September 2016.
PERSISTENT_START_INDEX = 31
PERSISTENT_LOS_MM = (
    0, 0, -2, -1, 0, 2, -1, 0, -3, -2, -1, 0, -1, -2, -5,
    -2, -1, -1, -3, -1, -2, -2, -1, -2, -2, 0, -5, -3, -4,
    -1, -5, -4, -6, -6, -8, -9, -10, -11, -13, -14, -14, -11, -8,
    -11, -18, -17, -17, -18, -19, -20, -19, -19, -20, -22, -20, -18, -22, -20,
)


def two_sided_cusum(
    residuals: np.ndarray, *, sigma: float = 5.0, reference: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate signed standardized residuals causally with zero resets."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive.')
    if not np.isfinite(reference) or reference < 0:
        raise ValueError('reference must be finite and non-negative.')
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
        raise ValueError('residuals must be a finite one-dimensional series.')
    positive = np.zeros(residuals.size)
    negative = np.zeros(residuals.size)
    upper = lower = 0.0
    for index, value in enumerate(residuals / sigma):
        upper = max(0.0, upper + value - reference)
        lower = max(0.0, lower - value - reference)
        positive[index], negative[index] = upper, lower
    return positive, negative


def normal_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return approximate dates, observed LOS, and expected normal baseline."""
    observed = np.asarray(NORMAL_LOS_MM, dtype=float)
    start, end = datetime(2015, 1, 1), datetime(2018, 3, 1)
    dates = np.array([
        start + timedelta(seconds=float(fraction) * (end - start).total_seconds())
        for fraction in np.linspace(0, 1, observed.size)
    ])
    return dates, observed, np.linspace(0.0, -5.0, observed.size)


def single_anomaly_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return case B's image estimates on the same approximate acquisition grid."""
    dates, _, baseline = normal_case()
    return dates, np.asarray(SINGLE_ANOMALY_LOS_MM, dtype=float), baseline


def persistent_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return case C's persistent deformation estimates with the shared baseline."""
    dates, _, baseline = normal_case()
    return dates, np.asarray(PERSISTENT_LOS_MM, dtype=float), baseline


def save_demo(
    output: Path, *, case: str = 'normal', fps: int = 4, dpi: int = 120
) -> None:
    """Save a causal GIF/MP4 or final PNG and the numeric trajectory as CSV."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    suffix = output.suffix.lower()
    if suffix not in {'.gif', '.mp4', '.png'}:
        raise ValueError('Output must end in .gif, .mp4, or .png.')
    if fps < 1 or dpi < 1:
        raise ValueError('fps and dpi must be positive.')
    if suffix == '.mp4' and not FFMpegWriter.isAvailable():
        raise RuntimeError('MP4 requires ffmpeg; use --output ending in .gif.')
    cases = {
        'normal': (normal_case, 'A. Normal behaviour — random variability, no persistent anomaly'),
        'single-anomaly': (single_anomaly_case, 'B. Single anomaly — one anomalous observation, no persistence'),
        'persistent': (persistent_case, 'C. Persistent deformation — repeated departures in the same direction'),
    }
    if case not in cases:
        raise ValueError(f'Unknown case: {case}')
    load_case, title = cases[case]
    dates, observed, baseline = load_case()
    positive, negative = two_sided_cusum(observed - baseline)
    statistic = np.maximum(positive, negative)
    threshold = 7.5
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 5.8))
    figure.subplots_adjust(left=0.08, right=0.91, top=0.78, bottom=0.22)
    cusum_axis = axis.twinx()
    axis.set(title=title, ylabel='LOS displacement (mm)')
    axis.set_xlim(dates[0], dates[-1])
    axis.set_ylim(-30, 12)
    cusum_axis.set(ylabel='CUSUM (dimensionless)',
                   ylim=(-1, max(30, float(statistic.max()) * 1.15)))
    # The two physical quantities use different scales on this shared timeline.
    axis.tick_params(axis='y', colors='#247fae')
    axis.yaxis.label.set_color('#247fae')
    axis.spines['left'].set_color('#247fae')
    cusum_axis.tick_params(axis='y', colors='#16853b')
    cusum_axis.yaxis.label.set_color('#16853b')
    cusum_axis.spines['right'].set_color('#16853b')
    cusum_axis.set_yticks(sorted({
        0.0, threshold, *np.arange(10.0, cusum_axis.get_ylim()[1], 10.0),
    }))
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axis.grid(alpha=0.18)
    band = axis.fill_between(dates, baseline - 10, baseline + 10,
                             color='#d9edfc', label='Expected normal range')
    baseline_line, = axis.plot(dates, baseline, '--', color='#7da9c4',
                               linewidth=1, label='Expected baseline')
    los_line, = axis.plot([], [], 'o-', color='#247fae', markersize=4,
                          label='Mean LOS')
    anomaly_indices = np.flatnonzero(np.abs(observed - baseline) > 10)
    anomaly_points, = axis.plot([], [], 'o', color='#d62728', markersize=7,
                                label='Level-1 anomalies', zorder=5)
    annotation = None
    if case == 'single-anomaly':
        anomaly_index = int(np.argmin(observed - baseline))
        annotation = axis.annotate(
            'Single anomaly (Level 1)',
            xy=(dates[anomaly_index], observed[anomaly_index]),
            xytext=(35, -8), textcoords='offset points', color='#d62728',
            fontsize=10, arrowprops={'arrowstyle': '->', 'color': '#d62728'},
        )
        annotation.set_visible(False)
    cusum_line, = cusum_axis.plot([], [], color='#20a44b', linewidth=2.5,
                                  label='CUSUM: max(C+, C−)')
    threshold_line = cusum_axis.axhline(threshold, color='#ed4b45',
                                       linestyle='--', label='CUSUM threshold h = 7.5 (right axis)')
    cusum_axis.annotate(
        'CUSUM h = 7.5 →', xy=(1, threshold),
        xycoords=cusum_axis.get_yaxis_transform(), xytext=(-7, 6),
        textcoords='offset points', ha='right', color='#ed4b45', fontsize=10,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 2},
    )
    crossings = np.flatnonzero(statistic >= threshold)
    crossing_index = int(crossings[0]) if crossings.size else None
    crossing_point, = cusum_axis.plot([], [], 'o', color='#d62728', markersize=8,
                                     zorder=6)
    crossing_annotation = None
    if crossing_index is not None:
        crossing_annotation = cusum_axis.annotate(
            'First threshold crossing\nLevel-2 warning',
            xy=(dates[crossing_index], statistic[crossing_index]),
            xytext=(-120, 32), textcoords='offset points', color='#d62728',
            fontsize=10, arrowprops={'arrowstyle': '->', 'color': '#d62728'},
        )
        crossing_annotation.set_visible(False)
    cursor = axis.axvline(dates[0], color='0.5', linewidth=0.8, alpha=0.5)
    status = figure.text(0.5, 0.12, '', ha='center', fontsize=11,
                         bbox={'facecolor': '#dff1df', 'edgecolor': 'none', 'pad': 8})
    figure.text(0.5, 0.03,
                'Approximate image transcription; evenly spaced dates. '
                'CUSUM computed from baseline residuals: σ = 5 mm, k = 0.5.',
                ha='center', fontsize=9, color='0.35')
    handles = [los_line, band, baseline_line, cusum_line, threshold_line]
    if case != 'normal':
        handles.append(anomaly_points)
    figure.legend(handles=handles, loc='upper center', ncol=3,
                  bbox_to_anchor=(0.5, 0.99), frameon=False)

    def update(index: int) -> tuple:
        end = index + 1
        los_line.set_data(dates[:end], observed[:end])
        cusum_line.set_data(dates[:end], statistic[:end])
        visible_anomalies = anomaly_indices[anomaly_indices <= index]
        anomaly_points.set_data(dates[visible_anomalies], observed[visible_anomalies])
        if annotation is not None:
            annotation.set_visible(index >= anomaly_index)
        if crossing_annotation is not None and crossing_index is not None:
            revealed = index >= crossing_index
            crossing_annotation.set_visible(revealed)
            crossing_point.set_data(
                [dates[crossing_index]] if revealed else [],
                [statistic[crossing_index]] if revealed else [],
            )
        cursor.set_xdata([dates[index], dates[index]])
        # Latch a warning once any revealed sample has crossed the threshold.
        warning = bool(np.any(statistic[:end] >= threshold))
        message = 'Persistent change — Level-2 warning' if warning else 'No persistent change (no warning)'
        if case == 'persistent' and not warning and index >= PERSISTENT_START_INDEX:
            message = 'Accumulating evidence; no Level-2 warning yet'
        if case == 'single-anomaly' and not warning and visible_anomalies.size:
            message = 'Level-1 anomaly; no Level-2 warning'
        status.set_text(f'{dates[index]:%Y-%m-%d}  |  CUSUM = {statistic[index]:.2f}  |  {message}')
        status.get_bbox_patch().set_facecolor('#f8d9d5' if warning else '#dff1df')
        return los_line, cusum_line, anomaly_points, crossing_point, cursor, status

    try:
        if suffix == '.png':
            update(len(dates) - 1)
            figure.savefig(output, dpi=dpi)
        else:
            animation = FuncAnimation(figure, update, frames=len(dates),
                                      interval=1000 / fps, repeat=False)
            writer = FFMpegWriter(fps=fps) if suffix == '.mp4' else PillowWriter(fps=fps)
            animation.save(output, writer=writer, dpi=dpi)
    finally:
        plt.close(figure)

    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('approximate_date', 'approximate_los_mm', 'baseline_mm',
                         'expected_lower_mm', 'expected_upper_mm', 'cusum_positive',
                         'cusum_negative', 'cusum', 'threshold', 'warning_latched',
                         'level1_anomaly'))
        for i, date in enumerate(dates):
            writer.writerow((date.date().isoformat(), observed[i], baseline[i],
                             baseline[i] - 10, baseline[i] + 10, positive[i], negative[i],
                             statistic[i], threshold, bool(np.any(statistic[:i + 1] >= threshold)),
                             bool(abs(observed[i] - baseline[i]) > 10)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('normal', 'single-anomaly', 'persistent'), default='normal',
                        help='Image-derived case: A (normal), B (single anomaly), or C (persistent).')
    parser.add_argument('--output', type=Path,
                        help='Defaults to outputs/level2_<case>.gif.')
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--dpi', type=int, default=120)
    args = parser.parse_args()
    if args.fps < 1 or args.dpi < 1:
        parser.error('--fps and --dpi must be positive.')
    if args.output is None:
        args.output = Path('outputs') / f'level2_{args.case.replace("-", "_")}.gif'
    save_demo(args.output, case=args.case, fps=args.fps, dpi=args.dpi)
    print(f'Wrote {args.output} and {args.output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
