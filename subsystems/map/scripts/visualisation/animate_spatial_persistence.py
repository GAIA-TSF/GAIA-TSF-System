"""Animate synthetic temporal persistence and spatial coherence.

python3 subsystems/map/scripts/animate_spatial_persistence.py \
    --output /tmp/spatial_persistence.mp4

The hand-drawn footprint is inspired by the supplied Cadia spatial summary;
coordinates are local metres, not georeferenced Cadia measurements. All time
series are synthetic. Yellow-to-red colours show downward baseline departures
(mm), not the source image's temporal range. Fixed scales and sampling locations
are used throughout. No CUSUM or operational risk calibration is implied.

A point must exceed 5 mm for three consecutive acquisitions. A coherent region
then requires eight or more such points connected through the grid's eight
neighbours. These are illustrative criteria, not calibrated warning settings.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np


ANOMALY_MM = 5.0
PERSISTENCE = 3
MIN_REGION_POINTS = 8
FRAME_COUNT = 48
MAX_CLUSTER_RADIUS_M = 450.0

# Approximate monitoring strip centreline, in a local coordinate frame.
OUTLINE = np.array([
    [620, 2950], [850, 2860], [1200, 3100], [1550, 3080],
    [1860, 3050], [2130, 3160], [2250, 2870], [2200, 2670],
    [2470, 2500], [2770, 2400], [2950, 2200], [2800, 1880],
    [2620, 1510], [2780, 1360], [3150, 1420], [3310, 1340],
    [3370, 1190], [3240, 900], [3160, 520], [2990, 280],
    [2810, 350], [2600, 520], [2300, 650], [2000, 610],
    [1630, 650], [1250, 730], [930, 900], [630, 1080],
    [430, 1390], [460, 1800], [530, 2160], [710, 2510],
    [740, 2750], [620, 2950],
], dtype=float)


def synthetic_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed grid, footprint mask and deterministic deformation frames."""
    x, y = np.meshgrid(np.arange(250, 3476, 35), np.arange(150, 3301, 35))
    distance = np.full(x.shape, np.inf)
    for start, end in zip(OUTLINE[:-1], OUTLINE[1:]):
        delta = end - start
        fraction = np.clip(((x-start[0])*delta[0] + (y-start[1])*delta[1])
                           / np.dot(delta, delta), 0, 1)
        distance = np.minimum(distance, np.hypot(
            x-start[0]-fraction*delta[0], y-start[1]-fraction*delta[1]))
    # Wider support on the southern and western embankments, as in the reference.
    width = np.where((y < 1150) & (x < 2400), 155.0, 75.0)
    mask = distance <= width
    rng = np.random.default_rng(17)
    spatial_texture = rng.uniform(0.9, 1.1, x.shape)
    frames = []
    for step in range(FRAME_COUNT):
        field = rng.uniform(0.0, 1.8, x.shape)
        # Short-lived isolated observations move each acquisition and cannot
        # satisfy the three-acquisition persistence requirement.
        support = np.argwhere(mask)
        selected = support[rng.choice(len(support), 7, replace=False)]
        field[selected[:, 0], selected[:, 1]] = rng.uniform(6, 10, len(selected))
        if step >= 12:
            progress = (step - 12) / (FRAME_COUNT - 13)
            radius = min(MAX_CLUSTER_RADIUS_M, 100 + 1000 * progress)
            strength = 7 + 23 * progress
            # A fixed core remains active while its connected neighbourhood grows.
            radial = np.hypot(x - 720, y - 1140)
            cluster = np.clip((radius + 100 - radial) / 140, 0, 1)
            cluster = np.where(radial <= radius, cluster, 0.0)
            field = np.maximum(field, strength * cluster * spatial_texture)
        frames.append(np.where(mask, np.clip(field, 0, 30), np.nan))
    return x, y, mask, np.asarray(frames)


def coherent_regions(persistent: np.ndarray) -> np.ndarray:
    """Keep eight-connected components containing at least MIN_REGION_POINTS."""
    remaining = set(map(tuple, np.argwhere(persistent)))
    coherent = np.zeros(persistent.shape, dtype=bool)
    while remaining:
        seed = remaining.pop()
        component = [seed]
        queue = deque([seed])
        while queue:
            row, column = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbour = (row + dr, column + dc)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        component.append(neighbour)
                        queue.append(neighbour)
        if len(component) >= MIN_REGION_POINTS:
            rows, columns = zip(*component)
            coherent[rows, columns] = True
    return coherent


def persistence_masks(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute causal consecutive exceedances and coherent regions per frame."""
    run = np.zeros(fields.shape[1:], dtype=int)
    runs, regions = [], []
    for field in fields:
        run = np.where(field > ANOMALY_MM, run + 1, 0)
        runs.append(run.copy())
        regions.append(coherent_regions(run >= PERSISTENCE))
    return np.asarray(runs), np.asarray(regions)


def save_animation(output: Path, *, fps: int = 4, dpi: int = 120) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    if output.suffix.lower() not in {'.mp4', '.gif', '.png'}:
        raise ValueError('Output must be MP4, GIF, or PNG.')
    if fps < 1 or dpi < 1:
        raise ValueError('fps and dpi must be positive.')
    if output.suffix.lower() == '.mp4' and not FFMpegWriter.isAvailable():
        raise RuntimeError('MP4 requires ffmpeg; use .gif instead.')
    x, y, mask, fields = synthetic_series()
    runs, regions = persistence_masks(fields)
    cmap = LinearSegmentedColormap.from_list('yellow_to_red', ['#ffe34d', '#ff9b23', '#ed431c', '#b50016'])
    fig, ax = plt.subplots(figsize=(9, 8.5), facecolor='#f6f9f9')
    fig.subplots_adjust(left=.12, right=.79, top=.82, bottom=.24)
    ax.set_facecolor('white')
    ax.set(xlim=(100, 3550), ylim=(0, 3400), aspect='equal',
           xlabel='Local easting (m)', ylabel='Local northing (m)')
    ax.grid(alpha=.16)
    ax.plot(OUTLINE[:, 0], OUTLINE[:, 1], color='#d8e1e1', lw=12, alpha=.35, zorder=1)
    ax.scatter(x[mask], y[mask], s=7, color='#d0dada', linewidths=0, zorder=2)
    points = ax.scatter([], [], c=[], s=11, cmap=cmap, norm=Normalize(5, 30), linewidths=0, zorder=3)
    cax = fig.add_axes([.83, .30, .025, .46])
    fig.colorbar(points, cax=cax, label='Downward departure from baseline (mm)', ticks=[5, 10, 15, 20, 25, 30])
    fig.suptitle('Spatial persistence of deformation', fontsize=21, color='#103b44', y=.96, weight='bold')
    phase = fig.text(.46, .895, '', ha='center', fontsize=14, color='#103b44')
    status = fig.text(.5, .145, '', ha='center', fontsize=11, color='#103b44')
    fig.text(.5, .077, 'Coherent region: ≥8 neighbouring points, each >5 mm for ≥3 consecutive acquisitions.',
             ha='center', fontsize=9, color='#526f77')
    fig.text(.5, .035, 'Synthetic illustration • Cadia-inspired footprint • Local coordinates • Fixed colour scale',
             ha='center', fontsize=9, color='#526f77')
    fig.legend(handles=[
        Line2D([], [], color='#d0dada', marker='o', ls='', label='Within expected range'),
        Line2D([], [], color='#71242b', ls='--', label='Persistent coherent region'),
    ], loc='center', bbox_to_anchor=(.5, .12), ncol=2, frameon=False, fontsize=9)
    contour = None

    def update(step: int) -> tuple:
        nonlocal contour
        active = mask & (fields[step] > ANOMALY_MM)
        points.set_offsets(np.column_stack((x[active], y[active])))
        points.set_array(fields[step][active])
        if contour is not None:
            contour.remove()
            contour = None
        if np.any(regions[step]):
            contour = ax.contour(x, y, regions[step].astype(float), levels=[.5],
                                 colors=['#71242b'], linewidths=1.3, linestyles='--', zorder=4)
        stage = 'Isolated, short-lived observations' if step < 12 else (
            'Persistent neighbouring deformation' if step < 25
            else 'Growth stops — smaller region persists and intensifies')
        phase.set_text(f'Acquisition {step + 1:02d} / {FRAME_COUNT} — {stage}')
        confirmed = np.count_nonzero(regions[step])
        status.set_text(f'Anomalous points: {np.count_nonzero(active)}   |   '
                        f'Persistent coherent points: {confirmed}\n'
                        + ('Repeated temporal evidence + coherent spatial behaviour' if confirmed
                           else 'No persistent coherent region yet'))
        return points, phase, status

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output.suffix.lower() == '.png':
            update(FRAME_COUNT - 1)
            fig.savefig(output, dpi=dpi)
        else:
            animation = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=1000/fps, repeat=False)
            writer = FFMpegWriter(fps=fps) if output.suffix.lower() == '.mp4' else PillowWriter(fps=fps)
            animation.save(output, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)
    np.savez_compressed(output.with_suffix('.npz'), local_easting=x, local_northing=y,
                        support_mask=mask, downward_departure_mm=fields,
                        consecutive_exceedances=runs, coherent_regions=regions,
                        anomaly_mm=ANOMALY_MM, persistence_acquisitions=PERSISTENCE,
                        minimum_region_points=MIN_REGION_POINTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/spatial_persistence.mp4'))
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--dpi', type=int, default=120)
    args = parser.parse_args()
    if args.fps < 1 or args.dpi < 1:
        parser.error('--fps and --dpi must be positive.')
    save_animation(args.output, fps=args.fps, dpi=args.dpi)
    print(f'Wrote {args.output} and {args.output.with_suffix(".npz")}')


if __name__ == '__main__':
    main()
