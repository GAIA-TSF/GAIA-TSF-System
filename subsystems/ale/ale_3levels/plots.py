"""Plots and animations from saved ALE outputs only (spec §6.1).

Every function reads engine output directories (``alerts.npz``,
``alerts.json``, ``history.csv``, ``sectors.csv``) and never recomputes a
metric or level, so enabling or disabling plots cannot change any number.
Written with matplotlib (Agg backend); animations as GIF (Pillow).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ale_3levels.outputs import get_metric, read_alerts

# Severity colours: NO_DATA, 0, 1, 2, 3 (WHITE/NONE ... RED/L3).
SEVERITY_COLOURS = ('#d9d9d9', '#ffffff', '#4daf4a', '#ffd92f', '#e41a1c')
SEVERITY_TICKS = (-1, 0, 1, 2, 3)


def _pyplot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def _severity_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(SEVERITY_COLOURS)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm


def _labels(metadata: dict[str, Any]) -> list[str]:
    mapping = metadata['severity_mapping']
    return [
        next(name for name, value in mapping.items() if value == code)
        for code in SEVERITY_TICKS
    ]


def _monitoring(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> np.ndarray:
    return arrays['dates'] >= metadata['input']['monitoring_start']


def _severity_map(arrays, index):
    severity = arrays['severity'][index].astype(float)
    return np.where(arrays['mask'], severity, np.nan)


def alert_map_frames(a_dir: Path, b_dir: Path, out_dir: Path) -> list[Path]:
    """One PNG per monitoring acquisition: engine A and engine B side by side."""
    plt = _pyplot()
    a, meta_a = read_alerts(a_dir)
    b, meta_b = read_alerts(b_dir)
    cmap, norm = _severity_cmap()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in np.flatnonzero(_monitoring(a, meta_a)):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
        for axis, arrays, metadata in ((axes[0], a, meta_a), (axes[1], b, meta_b)):
            image = axis.imshow(_severity_map(arrays, index), cmap=cmap, norm=norm)
            axis.set_title(f'{metadata["engine"]}  {arrays["dates"][index]}')
            axis.set_xticks([])
            axis.set_yticks([])
            bar = figure.colorbar(image, ax=axis, ticks=SEVERITY_TICKS, shrink=0.8)
            bar.ax.set_yticklabels(_labels(metadata))
        path = out_dir / f'alerts_{arrays["dates"][index]}.png'
        figure.savefig(path, dpi=90)
        plt.close(figure)
        paths.append(path)
    return paths


def alert_animation(a_dir: Path, b_dir: Path, path: Path, fps: int = 3) -> Path:
    """GIF of the side-by-side alert maps over the monitoring period."""
    plt = _pyplot()
    from matplotlib.animation import FuncAnimation, PillowWriter

    a, meta_a = read_alerts(a_dir)
    b, meta_b = read_alerts(b_dir)
    cmap, norm = _severity_cmap()
    frames = np.flatnonzero(_monitoring(a, meta_a))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    images = []
    for axis, arrays, metadata in ((axes[0], a, meta_a), (axes[1], b, meta_b)):
        image = axis.imshow(_severity_map(arrays, frames[0]), cmap=cmap, norm=norm)
        axis.set_xticks([])
        axis.set_yticks([])
        bar = figure.colorbar(image, ax=axis, ticks=SEVERITY_TICKS, shrink=0.8)
        bar.ax.set_yticklabels(_labels(metadata))
        images.append((axis, image, arrays, metadata))

    def update(index):
        for axis, image, arrays, metadata in images:
            image.set_data(_severity_map(arrays, index))
            axis.set_title(f'{metadata["engine"]}  {arrays["dates"][index]}')
        return [image for _, image, _, _ in images]

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return path


def three_level_history(b_dir: Path, path: Path) -> Path:
    """Engine B: TSF-wide L2 CUSUM and L3 regime-change score over time."""
    plt = _pyplot()
    arrays, metadata = read_alerts(b_dir)
    attrs = metadata['engine_attrs']
    mask = arrays['mask']
    dates = arrays['dates'].astype('datetime64[D]')

    def tsf_series(name):
        # TSF-level metrics are broadcast to every TSF pixel: take that value.
        return np.fmax.reduce(get_metric(arrays, metadata, name)[:, mask], axis=1)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    top.plot(dates, tsf_series('cusum_pos'), label='TSF L2 CUSUM (instability)')
    if 'l2_decision_threshold' in attrs:  # absent in outputs written before 0.1.0 B2
        top.axhline(attrs['l2_decision_threshold'], color='red', ls='--', label='h')
    top.set_ylabel('CUSUM (sigma units)')
    top.legend(loc='upper left')
    bottom.plot(
        dates,
        tsf_series('regime_change_score'),
        color='purple',
        label='regime_change_score',
    )
    if 'l3_watch_threshold' in attrs:
        bottom.axhline(
            attrs['l3_watch_threshold'], color='orange', ls='--', label='watch'
        )
        bottom.axhline(attrs['l3_alert_threshold'], color='red', ls='--', label='alert')
    bottom.set_ylim(0, 1)
    bottom.set_ylabel('L3 score')
    bottom.legend(loc='upper left')
    for reset in attrs.get('resets') or []:
        for axis in (top, bottom):
            axis.axvline(np.datetime64(reset['date']), color='grey', lw=0.8)
    figure.suptitle(f'{metadata["engine"]} history ({metadata["engine_version"]})')
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


def sector_timeline(a_dir: Path, b_dir: Path, path: Path) -> Path:
    """Sector severity over time for both engines (one panel per sector)."""
    plt = _pyplot()
    rows = {}
    for engine_dir in (a_dir, b_dir):
        _, metadata = read_alerts(engine_dir)
        with (engine_dir / 'sectors.csv').open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                rows.setdefault(row['sector_id'], {}).setdefault(
                    metadata['engine'], []
                ).append((np.datetime64(row['date']), int(row['severity'])))
    figure, axes = plt.subplots(
        len(rows), 1, figsize=(11, 2.6 * len(rows)), squeeze=False
    )
    for axis, (sector, engines) in zip(axes[:, 0], sorted(rows.items())):
        for engine, series in sorted(engines.items()):
            days, severity = zip(*series)
            axis.step(days, severity, where='post', label=engine)
        axis.set_yticks(SEVERITY_TICKS)
        axis.set_yticklabels(['NO_DATA', '0', '1', '2', '3'])
        axis.set_title(f'sector {sector}')
        axis.legend(loc='upper left')
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


def level_counts(engine_dir: Path, path: Path) -> Path:
    """Pixels per level and acquisition (history.csv), stacked."""
    plt = _pyplot()
    _, metadata = read_alerts(engine_dir)
    with (engine_dir / 'history.csv').open(encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    dates = np.array([np.datetime64(row['date']) for row in history])
    levels = metadata['levels']
    stacks = [[int(row[f'pixels_{label}']) for row in history] for label in levels]
    figure, axis = plt.subplots(figsize=(11, 4))
    colours = [SEVERITY_COLOURS[0], *SEVERITY_COLOURS[1:]][: len(levels)]
    axis.stackplot(
        dates, stacks, labels=levels, colors=colours, edgecolor='0.5', lw=0.3
    )
    axis.set_ylabel('pixels')
    axis.set_title(f'{metadata["engine"]}: pixels per level')
    axis.legend(loc='upper left', ncol=len(levels))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


def plot_run(
    run_dir: Path, out_dir: Path, *, maps: bool = True, animation: bool = False
) -> list[Path]:
    """All plots of a 'run --engine both' output directory."""
    a_dir, b_dir = Path(run_dir) / 'fixed_gates', Path(run_dir) / 'three_level'
    out_dir = Path(out_dir)
    paths = [
        three_level_history(b_dir, out_dir / 'three_level_history.png'),
        sector_timeline(a_dir, b_dir, out_dir / 'sector_timeline.png'),
        level_counts(a_dir, out_dir / 'fixed_gates_level_counts.png'),
        level_counts(b_dir, out_dir / 'three_level_level_counts.png'),
    ]
    if maps:
        paths += alert_map_frames(a_dir, b_dir, out_dir / 'maps')
    if animation:
        paths.append(alert_animation(a_dir, b_dir, out_dir / 'alerts.gif'))
    return paths
