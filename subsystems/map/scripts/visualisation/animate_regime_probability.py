"""Animate an illustrative Level-3 regime-change probability trajectory.

python3 subsystems/map/scripts/animate_regime_probability.py \
    --output /tmp/regime_probability.mp4

Monthly LOS and probability values are approximate visual transcriptions of
the supplied schematic (January 2015 to February 2018). Probabilities are
illustrative input values, not estimates from a fitted probabilistic model.
Phase boundaries follow the schematic, not probability alarm thresholds.
PNG exports the final frame; MP4/GIF reveal paired observations causally.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np


LOS_MM = (
    -3, -3, -3.3, -3.5, -4, -4.5, -5, -5.7, -6.5, -7.2, -8, -8.8,
    -9.5, -10, -10.5, -11.5, -12.5, -13.5, -14.7, -16, -17.3, -18.8,
    -20.5, -22.5, -25, -27.5, -30.5, -33.5, -36.5, -40.5, -44,
    -47.5, -50.3, -53, -55.5, -58, -61, -64,
)
PROBABILITY = (
    .01, .01, .01, .01, .01, .012, .012, .013, .015, .017, .02, .025,
    .03, .035, .043, .05, .06, .069, .078, .088, .10, .115, .132,
    .16, .20, .25, .32, .41, .53, .74, .88, .95, .975, .988,
    .995, .997, .998, .998,
)
PHASE_STARTS = (0, 12, 26)
PHASE_NAMES = ('Normal behaviour', 'Increasing evidence', 'Regime change')
PHASE_DETAILS = ('No persistent anomaly', 'Persistent anomalies', 'High probability develops')
PHASE_COLORS = ('#e5f3fd', '#fff3d7', '#fce4e4')
PHASE_INKS = ('#185780', '#8c5b0d', '#a9242b')


def reference_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return explicitly approximate monthly dates and paired image estimates."""
    dates = np.array([datetime(2015 + i // 12, i % 12 + 1, 1) for i in range(len(LOS_MM))])
    return dates, np.asarray(LOS_MM, dtype=float), np.asarray(PROBABILITY, dtype=float)


def phase_index(index: int) -> int:
    return int(np.searchsorted(PHASE_STARTS, index, side='right') - 1)


def save_animation(output: Path, *, fps: int = 4, dpi: int = 120) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.ticker import PercentFormatter

    suffix = output.suffix.lower()
    if suffix not in {'.mp4', '.gif', '.png'}:
        raise ValueError('Output must be MP4, GIF, or PNG.')
    if fps < 1 or dpi < 1:
        raise ValueError('fps and dpi must be positive.')
    if suffix == '.mp4' and not FFMpegWriter.isAvailable():
        raise RuntimeError('MP4 requires ffmpeg; use .gif instead.')
    dates, los, probability = reference_data()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.35, 1]})
    fig.set_facecolor('#f6f9f9')
    fig.subplots_adjust(left=.11, right=.97, bottom=.18, top=.79, hspace=.20)
    fig.suptitle('From persistent deformation to regime change', y=.975,
                 fontsize=22, color='#103b44', weight='bold')
    phase_status = fig.text(.54, .91, '', ha='center', color='#103b44', fontsize=13)
    boundaries = [dates[i] for i in PHASE_STARTS] + [dates[-1]]
    for ax in axes:
        for phase in range(3):
            ax.axvspan(boundaries[phase], boundaries[phase + 1],
                       color=PHASE_COLORS[phase], zorder=0)
        for boundary in boundaries[1:-1]:
            ax.axvline(boundary, color='#a6b3b6', linestyle='--', linewidth=1)
        ax.set_xlim(dates[0], dates[-1])
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    for phase in range(3):
        middle = boundaries[phase] + (boundaries[phase+1] - boundaries[phase]) / 2
        axes[0].text(middle, 1.04,
                     f'{phase+1}. {PHASE_NAMES[phase]}\n{PHASE_DETAILS[phase]}',
                     transform=axes[0].get_xaxis_transform(), ha='center', va='bottom',
                     fontsize=12, color=PHASE_INKS[phase], weight='bold')
    axes[0].set(ylabel='Cumulative LOS displacement (mm)', ylim=(-70, 10))
    axes[1].set(ylabel='P(regime change)', ylim=(-.025, 1.08))
    axes[1].set_title('Regime-change probability · Level 3', loc='left',
                       color='#103b44', fontsize=13, weight='bold', pad=10)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    los_line, = axes[0].plot([], [], 'o-', color='#176899', markersize=4, linewidth=2)
    probability_line, = axes[1].plot([], [], 'o-', color='#c8242c', markersize=4, linewidth=2)
    cursors = [ax.axvline(dates[0], color='#506a73', alpha=.5, linewidth=1) for ax in axes]
    notes = []
    for phase, index, text, position in (
        (0, 6, 'Small, stable deformation', (.19, .32)),
        (1, 21, 'Persistent departure\nEvidence accumulates', (.48, .30)),
        (2, 32, 'Accelerating deformation\nRegime-change evidence', (.83, .67)),
    ):
        note = axes[0].annotate(
            text, xy=(dates[index], los[index]), xytext=position, textcoords='axes fraction',
            ha='center', fontsize=10, color=PHASE_INKS[phase],
            bbox={'boxstyle': 'round,pad=.6', 'facecolor': PHASE_COLORS[phase],
                  'edgecolor': PHASE_INKS[phase], 'alpha': .95},
            arrowprops={'arrowstyle': '->', 'color': PHASE_INKS[phase]},
        )
        note.set_visible(False)
        notes.append((index, note))
    status = fig.text(.54, .10, '', ha='center', fontsize=12, color='#103b44')
    fig.text(.54, .04,
             'Approximate image transcription • Illustrative probabilities, not model-calibrated estimates',
             ha='center', fontsize=10, color='#607a82')

    def update(index: int) -> tuple:
        end = index + 1
        los_line.set_data(dates[:end], los[:end])
        probability_line.set_data(dates[:end], probability[:end])
        for cursor in cursors:
            cursor.set_xdata([dates[index], dates[index]])
        for reveal, note in notes:
            note.set_visible(index >= reveal)
        phase = phase_index(index)
        phase_status.set_text(f'{dates[index]:%B %Y}  |  {PHASE_NAMES[phase]}')
        phase_status.set_color(PHASE_INKS[phase])
        status.set_text(f'LOS: {los[index]:.1f} mm    |    P(regime change): {probability[index]:.1%}')
        return los_line, probability_line, *cursors, phase_status, status

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if suffix == '.png':
            update(len(dates)-1)
            fig.savefig(output, dpi=dpi)
        else:
            animation = FuncAnimation(fig, update, frames=len(dates), interval=1000/fps, repeat=False)
            writer = FFMpegWriter(fps=fps) if suffix == '.mp4' else PillowWriter(fps=fps)
            animation.save(output, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)
    with output.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('approximate_date', 'approximate_cumulative_los_mm',
                         'illustrative_regime_probability', 'schematic_phase'))
        for index, date in enumerate(dates):
            writer.writerow((date.date().isoformat(), los[index], probability[index],
                             PHASE_NAMES[phase_index(index)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/regime_probability.mp4'))
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--dpi', type=int, default=120)
    args = parser.parse_args()
    if args.fps < 1 or args.dpi < 1:
        parser.error('--fps and --dpi must be positive.')
    save_animation(args.output, fps=args.fps, dpi=args.dpi)
    print(f'Wrote {args.output} and {args.output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
