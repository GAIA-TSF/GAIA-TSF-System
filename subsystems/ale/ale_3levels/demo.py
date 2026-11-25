"""Small simulated MAP output for tests and quick end-to-end runs.

Writes a MAP-format lead-1 residual table (schema 2.0) for a small grid:
cumulative LOS in mm with a stable trend, Gaussian noise, one 24-day gap in
the 12-day cadence, and a square patch that accelerates away from the
satellite (negative LOS) from ``failure_start``. The "MAP forecast" is
persistence plus the stable trend, so residuals are noise plus the failure.

:func:`write_demo_site` adds a TSF mask, the ground-truth failure mask (for
the comparison report only) and an ALE configuration, laid out like a real
site::

    <out>/results/forecasts/slope_trf/residuals.npz|.json
    <out>/static/tsf_mask.tif
    <out>/static/roi_failure_mask.tif      (ground truth; compare --truth only)
    <out>/ale_demo.yaml
    <out>/README.txt

Simulated data, not a physical scenario. Deterministic for a given seed.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

TABLE_COLUMNS = (
    'series_id',
    'row',
    'col',
    'x',
    'y',
    'forecast_origin_date',
    'lead',
    'target_date',
    'target_date_status',
    'last_observation_date',
    'persistence_value',
    'predicted_value',
    'prediction_lower',
    'prediction_upper',
    'period',
    'sample_role',
    'observed_value',
    'residual',
)
ORIGIN_X, ORIGIN_Y, PIXEL = 500000.0, 7000000.0, 10.0
CRS = 'EPSG:32735'
TREND_MM_PER_DAY = -0.02


def acquisition_offsets(n_dates: int) -> np.ndarray:
    """Day offsets: 12-day cadence with one 24-day gap (after the 10th date)."""
    return np.cumsum([0] + [24 if i == 10 else 12 for i in range(1, n_dates)])


def simulate_series(
    *,
    n_dates: int,
    height: int,
    width: int,
    seed: int,
    patch: tuple[slice, slice],
    failure_start: int,
    failure_rate_mm: float = 0.15,
    noise_mm: float = 0.8,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return ``(dates, day offsets, cumulative LOS mm (time, row, col))``."""
    rng = np.random.default_rng(seed)
    offsets = acquisition_offsets(n_dates)
    dates = [(date(2020, 1, 1) + timedelta(days=int(d))).isoformat() for d in offsets]
    trend = TREND_MM_PER_DAY * offsets[:, None, None] * np.ones((1, height, width))
    noise = rng.normal(0.0, noise_mm, size=(n_dates, height, width))
    series = trend + noise
    extra = np.zeros(n_dates)
    for index in range(failure_start, n_dates):
        extra[index] = -failure_rate_mm * (index - failure_start + 1) ** 2
    series[:, patch[0], patch[1]] += extra[:, None, None]
    return dates, offsets, series


def write_residual_table(
    directory: Path,
    *,
    n_dates: int = 48,
    height: int = 12,
    width: int = 12,
    monitoring_index: int = 28,
    seed: int = 7,
    patch: tuple[slice, slice] = (slice(3, 9), slice(3, 9)),
    failure_start: int = 34,
    failure_rate_mm: float = 0.15,
    noise_mm: float = 0.8,
    mask: np.ndarray | None = None,
    model_version: str = 'slope_trf@long',
) -> Path:
    """Write ``residuals.npz`` + ``residuals.json`` (MAP table schema 2.0).

    Rows exist only for pixels in ``mask`` (default: all pixels). Sample roles
    of calibration-period rows: first 60 % train, next 20 % validation, last
    20 % test (temporal split).
    """
    dates, offsets, series = simulate_series(
        n_dates=n_dates,
        height=height,
        width=width,
        seed=seed,
        patch=patch,
        failure_start=failure_start,
        failure_rate_mm=failure_rate_mm,
        noise_mm=noise_mm,
    )
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    columns: dict[str, list] = {name: [] for name in TABLE_COLUMNS}
    monitoring_start = dates[monitoring_index]
    for origin in range(n_dates - 1):
        target = origin + 1
        dt = offsets[target] - offsets[origin]
        in_monitoring = dates[target] >= monitoring_start
        role = 'none'
        if not in_monitoring:
            fraction = target / monitoring_index
            role = (
                'train'
                if fraction < 0.6
                else 'validation'
                if fraction < 0.8
                else 'test'
            )
        for row in range(height):
            for col in range(width):
                if not mask[row, col]:
                    continue
                persistence = series[origin, row, col]
                predicted = persistence + TREND_MM_PER_DAY * dt
                observed = series[target, row, col]
                _append_row(
                    columns,
                    row=row,
                    col=col,
                    width=width,
                    origin=dates[origin],
                    target=dates[target],
                    persistence=persistence,
                    predicted=predicted,
                    observed=observed,
                    period='monitoring' if in_monitoring else 'calibration',
                    role=role,
                )
    arrays = {name: np.asarray(values) for name, values in columns.items()}
    sidecar = {
        'schema_version': '2.0',
        'model': 'trf',
        'model_version': model_version,
        'target_variable': 'cumulative_los',
        'target_feature': 'cumulative_los_current',
        'target_units': 'mm',
        'horizon': 1,
        'calibration_window': [dates[0], dates[monitoring_index - 1]],
        'monitoring_window': [monitoring_start, dates[-1]],
        'grid': {
            'crs': CRS,
            'transform': [PIXEL, 0.0, ORIGIN_X, 0.0, -PIXEL, ORIGIN_Y],
            'height': height,
            'width': width,
        },
        'product': 'map_forecast_residuals',
        'description': 'Simulated by ale_3levels.demo (not a MAP run).',
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'residuals.npz'
    np.savez_compressed(path, **arrays)
    path.with_suffix('.json').write_text(
        json.dumps(sidecar, indent=2), encoding='utf-8'
    )
    return path


def _append_row(
    columns,
    *,
    row,
    col,
    width,
    origin,
    target,
    persistence,
    predicted,
    observed,
    period,
    role,
):
    columns['series_id'].append(row * width + col)
    columns['row'].append(row)
    columns['col'].append(col)
    columns['x'].append(ORIGIN_X + PIXEL * (col + 0.5))
    columns['y'].append(ORIGIN_Y - PIXEL * (row + 0.5))
    columns['forecast_origin_date'].append(origin)
    columns['lead'].append(1)
    columns['target_date'].append(target)
    columns['target_date_status'].append('acquired')
    columns['last_observation_date'].append(origin)
    columns['persistence_value'].append(persistence)
    columns['predicted_value'].append(predicted)
    columns['prediction_lower'].append(predicted - 1.0)
    columns['prediction_upper'].append(predicted + 1.0)
    columns['period'].append(period)
    columns['sample_role'].append(role)
    columns['observed_value'].append(observed)
    columns['residual'].append(observed - predicted)


def _write_mask(path: Path, mask: np.ndarray) -> Path:
    import rasterio
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype='uint8',
        crs=CRS,
        transform=Affine(PIXEL, 0.0, ORIGIN_X, 0.0, -PIXEL, ORIGIN_Y),
    ) as raster:
        raster.write(mask.astype('uint8'), 1)
    return path


def write_demo_site(
    out_dir: Path,
    *,
    size: int = 24,
    n_dates: int = 60,
    monitoring_index: int = 30,
    failure_start: int = 32,
    failure_rate_mm: float = 0.4,
    noise_mm: float = 0.8,
    seed: int = 7,
    base_config: Path | None = None,
) -> dict[str, Any]:
    """Write a complete small demo site (table, masks, ALE config, README)."""
    from ale_3levels.config import DEFAULT_CONFIG

    if size < 16:
        raise ValueError('size must be at least 16 pixels (control area needs room).')
    out_dir = Path(out_dir)
    third = size // 4
    patch = (slice(third, third + size // 4), slice(third, third + size // 4))
    mask = np.ones((size, size), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = (
        False  # border outside the TSF
    )
    table = write_residual_table(
        out_dir / 'results' / 'forecasts' / 'slope_trf',
        n_dates=n_dates,
        height=size,
        width=size,
        monitoring_index=monitoring_index,
        seed=seed,
        patch=patch,
        failure_start=failure_start,
        failure_rate_mm=failure_rate_mm,
        noise_mm=noise_mm,
        mask=mask,
        model_version='slope_trf@demo',
    )
    truth = np.zeros_like(mask)
    truth[patch] = True
    tsf_mask = _write_mask(out_dir / 'static' / 'tsf_mask.tif', mask)
    truth_mask = _write_mask(out_dir / 'static' / 'roi_failure_mask.tif', truth)
    config = out_dir / 'ale_demo.yaml'
    base = Path(base_config or DEFAULT_CONFIG).resolve()
    config.write_text(
        '# ALE configuration for the simulated demo site (ale_3levels.demo).\n'
        f'base: {base}\n'
        'input:\n'
        '  residuals: results/forecasts/slope_trf/residuals.npz\n'
        '  mask_file: static/tsf_mask.tif\n'
        '  model_id: trf\n'
        'site:\n'
        '  site_id: ale_demo\n'
        'spatial:\n'
        '  # Small grid: smaller coherent areas than the default.\n'
        '  min_pixels: 9\n'
        '  coherence:\n'
        '    minimum_area_pixels: 9\n',
        encoding='utf-8',
    )
    readme = out_dir / 'README.txt'
    readme.write_text(
        'Simulated ALE demo site (written by ale_3levels.demo; not a MAP run).\n\n'
        f'grid {size}x{size} px (border outside the TSF), {n_dates} acquisitions '
        f'(12-day cadence, one 24-day gap), monitoring from acquisition {monitoring_index}.\n'
        f'Failure patch rows/cols {patch[0].start}-{patch[0].stop - 1}, accelerating away '
        f'from the satellite from acquisition {failure_start}; noise sigma {noise_mm} mm; '
        f'seed {seed}.\n\n'
        'Run from subsystems/ale of the repository:\n'
        f'  python3 -m ale_3levels run --engine both --config {config} --out {out_dir / "runs/demo"}\n'
        f'  python3 -m ale_3levels compare --a {out_dir / "runs/demo/fixed_gates"} '
        f'--b {out_dir / "runs/demo/three_level"} --truth {truth_mask} --out {out_dir / "report"}\n',
        encoding='utf-8',
    )
    return {
        'table': table,
        'tsf_mask': tsf_mask,
        'truth_mask': truth_mask,
        'config': config,
        'readme': readme,
    }
