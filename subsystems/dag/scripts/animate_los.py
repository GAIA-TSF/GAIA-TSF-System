"""Animate the complete Sentinel-1 LOS observation period.

Example:
python3 subsystems/dag/scripts/animate_los.py \
    --config subsystems/dag/config.yaml \
    --fps 4 \
    --dpi 150
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import array_bounds
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.dag.plugins.ingestion.sentinel1_loader import Sentinel1LOSLoader
from subsystems.dag.utils.raster import apply_mask


LOGGER = logging.getLogger(__name__)


def symmetric_color_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    """Return a robust, positive limit shared by every animation frame."""
    if not 0 < percentile <= 100:
        raise ValueError('color percentile must be in (0, 100].')
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError('LOS stack contains no finite values.')
    limit = float(np.percentile(finite, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.max(finite))
    return limit if limit > 0 else 1.0


def load_animation_inputs(
    config_path: Path,
) -> tuple[np.ndarray, tuple[Any, ...], Any, str, Path]:
    """Load the configured LOS stack, TSF mask, unit, and default output path."""
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise TypeError('DAG config must be a mapping.')
    project_value = config.get('project_dir')
    if not isinstance(project_value, str) or not project_value.strip():
        raise KeyError('DAG config requires project_dir.')
    project_dir = Path(project_value).expanduser().resolve()
    scenario = config.get('slope_stability')
    if not isinstance(scenario, dict):
        raise KeyError('Missing slope_stability configuration.')
    inputs = scenario.get('inputs')
    static = scenario.get('static')
    results = scenario.get('results')
    if not isinstance(inputs, dict) or not isinstance(static, dict):
        raise KeyError('Slope LOS inputs and static mask must be configured.')
    los = inputs.get('los')
    if not isinstance(los, dict):
        raise KeyError('Missing slope_stability.inputs.los configuration.')

    los_directory = _resolve(project_dir, los['directory'])
    mask_path = _resolve(project_dir, static['tsf_mask'])
    series = Sentinel1LOSLoader().load(
        los_directory,
        str(los['filename_pattern']),
    )
    with rasterio.open(mask_path) as source:
        mask = source.read(1).astype(bool)
        mask_signature = (source.crs, source.transform, source.height, source.width)
    series_signature = (
        series.profile.crs,
        series.profile.transform,
        series.profile.height,
        series.profile.width,
    )
    if mask_signature != series_signature:
        raise ValueError('TSF mask grid does not match the LOS raster grid.')
    if not np.any(mask):
        raise ValueError('TSF mask contains no selected pixels.')

    eda = results.get('eda', {}) if isinstance(results, dict) else {}
    unit = str(eda.get('displacement_unit', '')).strip() if isinstance(eda, dict) else ''
    default_output = project_dir / 'results' / 'eda' / 'los_observation_period.mp4'
    return (
        apply_mask(series.data, mask),
        series.dates,
        series.profile,
        unit,
        default_output,
    )


def save_los_animation(
    values: np.ndarray,
    dates: tuple[Any, ...],
    profile: Any,
    output_path: Path,
    *,
    unit: str = '',
    fps: int = 4,
    dpi: int = 150,
    color_percentile: float = 98.0,
    color_limit: float | None = None,
) -> Path:
    """Write a fixed-scale LOS animation as MP4 or GIF."""
    if values.ndim != 3 or values.shape[0] != len(dates):
        raise ValueError('LOS animation stack must match the acquisition dates.')
    if fps < 1 or dpi < 1:
        raise ValueError('fps and dpi must be positive.')
    limit = (
        symmetric_color_limit(values, color_percentile)
        if color_limit is None
        else float(color_limit)
    )
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('color_limit must be finite and positive.')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    bounds = array_bounds(profile.height, profile.width, profile.transform)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(
        values[0],
        cmap='RdBu',
        vmin=-limit,
        vmax=limit,
        extent=extent,
        origin='upper',
    )
    unit_suffix = f' [{unit}]' if unit else ''
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(f'LOS displacement{unit_suffix}')
    axis.set_xlabel('Easting')
    axis.set_ylabel('Northing')
    title = axis.set_title('')

    def update(frame_index: int) -> tuple[Any, Any]:
        image.set_data(values[frame_index])
        title.set_text(
            'Sentinel-1 LOS displacement — '
            f'{dates[frame_index].isoformat()} '
            f'({frame_index + 1}/{len(dates)})'
        )
        return image, title

    animation = FuncAnimation(
        figure,
        update,
        frames=len(dates),
        interval=1000.0 / fps,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = output_path
    if output_path.suffix.lower() == '.gif':
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        try:
            animation.save(output_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
        except (FileNotFoundError, RuntimeError, subprocess.SubprocessError):
            saved_path = output_path.with_suffix('.gif')
            LOGGER.warning('ffmpeg unavailable; writing %s instead.', saved_path)
            animation.save(saved_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    return saved_path


def _resolve(project_dir: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Animate all configured Sentinel-1 LOS acquisitions.',
    )
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--color-percentile', type=float, default=98.0)
    parser.add_argument('--color-limit', type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    values, dates, profile, unit, default_output = load_animation_inputs(
        args.config.expanduser().resolve()
    )
    output = default_output if args.output is None else args.output.expanduser().resolve()
    saved = save_los_animation(
        values,
        dates,
        profile,
        output,
        unit=unit,
        fps=args.fps,
        dpi=args.dpi,
        color_percentile=args.color_percentile,
        color_limit=args.color_limit,
    )
    print(
        {
            'frames': len(dates),
            'start_date': dates[0].isoformat(),
            'end_date': dates[-1].isoformat(),
            'output': str(saved),
        }
    )


if __name__ == '__main__':
    main()
