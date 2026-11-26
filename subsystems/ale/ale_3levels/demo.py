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
    model_id: str = 'trf',
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
        'model': model_id,
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


# ---------------------------------------------------------------------------
# Showcase site: a larger scenario for demonstrations of both engines.
# ---------------------------------------------------------------------------

#: Zone codes of ``static/demo_zones.tif`` (0 = outside the TSF).
SHOWCASE_ZONES = {'outside': 0, 'stable': 1, 'creep': 2, 'failure': 3}
SHOWCASE_START = date(2019, 1, 5)


def showcase_layout(height: int = 60, width: int = 80) -> dict[str, Any]:
    """Masks of the showcase site, laid out relative to the grid size.

    An elliptical TSF; a consolidating ("creep") area on the west beach; an
    accelerating failure on the southern dam wall; sectors ``west_beach``,
    ``east_beach`` and ``dam_wall``. The stable east beach is the farthest
    part of the TSF from the failure, where the comparison report places its
    control area.
    """
    rows, cols = np.indices((height, width))
    mask = ((rows + 0.5 - height / 2) / (0.45 * height)) ** 2 + (
        (cols + 0.5 - width / 2) / (0.46 * width)
    ) ** 2 <= 1.0

    def box(r0, r1, c0, c1):
        return (
            (rows >= int(r0 * height))
            & (rows < int(r1 * height))
            & (cols >= int(c0 * width))
            & (cols < int(c1 * width))
            & mask
        )

    creep = box(0.20, 0.43, 0.18, 0.40)
    failure = box(0.70, 0.87, 0.42, 0.58)
    zones = np.where(mask, SHOWCASE_ZONES['stable'], SHOWCASE_ZONES['outside'])
    zones[creep] = SHOWCASE_ZONES['creep']
    zones[failure] = SHOWCASE_ZONES['failure']
    split_row, split_col = int(0.65 * height), int(0.5 * width)
    sectors = {
        'west_beach': (0, split_row, 0, split_col),
        'east_beach': (0, split_row, split_col, width),
        'dam_wall': (split_row, height, 0, width),
    }
    return {
        'mask': mask,
        'zones': zones.astype('uint8'),
        'creep': creep,
        'failure': failure,
        'sectors': sectors,
        'aps_centre': (0.45 * height, 0.72 * width),
        'aps_sigma': 0.12 * width,
    }


def simulate_showcase(
    *,
    height: int = 60,
    width: int = 80,
    n_calibration: int = 110,
    n_monitoring: int = 60,
    seed: int = 11,
    noise_mm: float = 1.0,
    stable_rate_mm_per_day: float = -0.02,
    creep_rate_mm_per_day: float = -0.25,
    failure_onset: int = 12,
    failure_initial_rate_mm_per_day: float = 0.2,
    failure_lead_days: int = 30,
    atmosphere_mm: float = 1.5,
    aps_date: int = 6,
    aps_amplitude_mm: float = 8.0,
) -> dict[str, Any]:
    """Cumulative LOS (mm) and the simulated MAP forecast of the showcase site.

    * Stable ground and the creep area move at constant rates (the creep
      area at about 7.5 mm per 30 days). The simulated MAP has learned each
      pixel's rate during calibration: forecast = persistence + rate * dt.
    * The failure area accelerates away from the satellite from monitoring
      acquisition ``failure_onset``, following an inverse-velocity
      (Fukuzono) law: the extra velocity is v0 * T / (t_f - t), with the
      projected failure ``failure_lead_days`` after the last acquisition.
      MAP does not foresee the acceleration.
    * Every acquisition carries a smooth random atmospheric field (a ramp
      plus one broad blob, each of typical size ``atmosphere_mm``), in
      calibration and monitoring alike.
    * One monitoring acquisition (``aps_date``) carries an extra strong
      atmospheric artefact (a one-date bump of ``aps_amplitude_mm``).
    * Independent Gaussian noise of ``noise_mm`` per pixel.
    """
    layout = showcase_layout(height, width)
    n_dates = n_calibration + n_monitoring
    offsets = acquisition_offsets(n_dates).astype(float)
    dates = [(SHOWCASE_START + timedelta(days=int(d))).isoformat() for d in offsets]
    rate = np.full((height, width), stable_rate_mm_per_day)
    rate[layout['creep']] = creep_rate_mm_per_day
    rng = np.random.default_rng(seed)
    series = rate[None] * offsets[:, None, None]
    series = series + rng.normal(0.0, noise_mm, size=(n_dates, height, width))

    onset = n_calibration + failure_onset
    t0, t_end = offsets[onset], offsets[-1] + failure_lead_days
    span = t_end - t0
    after = np.clip(offsets - t0, 0.0, None)
    extra = -failure_initial_rate_mm_per_day * span * np.log(span / (span - after))
    series[:, layout['failure']] += extra[:, None]

    rows, cols = np.indices((height, width))
    u, v = rows / height - 0.5, cols / width - 0.5
    for index in range(n_dates):
        ramp_r, ramp_c, blob = rng.normal(0.0, atmosphere_mm, size=3)
        centre = rng.uniform(0.0, 1.0, size=2)
        radius = rng.uniform(0.15, 0.35)
        series[index] += ramp_r * u + ramp_c * v
        series[index] += blob * np.exp(
            -((u + 0.5 - centre[0]) ** 2 + (v + 0.5 - centre[1]) ** 2) / (2 * radius**2)
        )
    centre_r, centre_c = layout['aps_centre']
    bump = aps_amplitude_mm * np.exp(
        -((rows - centre_r) ** 2 + (cols - centre_c) ** 2)
        / (2 * layout['aps_sigma'] ** 2)
    )
    series[n_calibration + aps_date] += bump
    return {
        'dates': dates,
        'offsets': offsets,
        'series': series,
        'rate': rate,
        'layout': layout,
        'monitoring_index': n_calibration,
        'failure_onset_date': dates[onset],
        'aps_date': dates[n_calibration + aps_date],
    }


def write_showcase_table(
    directory: Path, simulation: dict[str, Any], *, model_version: str
) -> Path:
    """Vectorised MAP table (schema 2.0) for :func:`simulate_showcase`."""
    dates = np.asarray(simulation['dates'])
    offsets = simulation['offsets']
    series = simulation['series']
    mask = simulation['layout']['mask']
    monitoring_index = simulation['monitoring_index']
    height, width = mask.shape
    pixel_rows, pixel_cols = np.nonzero(mask)
    n_pixels, n_origins = pixel_rows.size, len(dates) - 1

    origin = np.repeat(np.arange(n_origins), n_pixels)
    target = origin + 1
    row = np.tile(pixel_rows, n_origins)
    col = np.tile(pixel_cols, n_origins)
    dt = offsets[target] - offsets[origin]
    persistence = series[origin, row, col]
    predicted = persistence + simulation['rate'][row, col] * dt
    observed = series[target, row, col]
    monitoring = target >= monitoring_index
    fraction = target / monitoring_index
    role = np.where(
        monitoring,
        'none',
        np.where(
            fraction < 0.6, 'train', np.where(fraction < 0.8, 'validation', 'test')
        ),
    )
    arrays = {
        'series_id': row * width + col,
        'row': row,
        'col': col,
        'x': ORIGIN_X + PIXEL * (col + 0.5),
        'y': ORIGIN_Y - PIXEL * (row + 0.5),
        'forecast_origin_date': dates[origin],
        'lead': np.ones_like(origin),
        'target_date': dates[target],
        'target_date_status': np.full(origin.shape, 'acquired'),
        'last_observation_date': dates[origin],
        'persistence_value': persistence,
        'predicted_value': predicted,
        'prediction_lower': predicted - 1.0,
        'prediction_upper': predicted + 1.0,
        'period': np.where(monitoring, 'monitoring', 'calibration'),
        'sample_role': role,
        'observed_value': observed,
        'residual': observed - predicted,
    }
    sidecar = {
        'schema_version': '2.0',
        'model': 'trf',
        'model_version': model_version,
        'target_variable': 'cumulative_los',
        'target_feature': 'cumulative_los_current',
        'target_units': 'mm',
        'horizon': 1,
        'calibration_window': [str(dates[0]), str(dates[monitoring_index - 1])],
        'monitoring_window': [str(dates[monitoring_index]), str(dates[-1])],
        'grid': {
            'crs': CRS,
            'transform': [PIXEL, 0.0, ORIGIN_X, 0.0, -PIXEL, ORIGIN_Y],
            'height': height,
            'width': width,
        },
        'product': 'map_forecast_residuals',
        'description': 'Simulated by ale_3levels.demo showcase (not a MAP run).',
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'residuals.npz'
    np.savez_compressed(path, **{name: arrays[name] for name in TABLE_COLUMNS})
    path.with_suffix('.json').write_text(
        json.dumps(sidecar, indent=2), encoding='utf-8'
    )
    return path


def _write_raster(path: Path, values: np.ndarray) -> Path:
    import rasterio
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype='uint8',
        crs=CRS,
        transform=Affine(PIXEL, 0.0, ORIGIN_X, 0.0, -PIXEL, ORIGIN_Y),
    ) as raster:
        raster.write(values.astype('uint8'), 1)
    return path


def _write_sectors(path: Path, sectors: dict[str, tuple[int, int, int, int]]) -> Path:
    """Sector rectangles (row/col bounds) as GeoJSON in the grid CRS."""
    features = []
    for name, (r0, r1, c0, c1) in sectors.items():
        x0, x1 = ORIGIN_X + PIXEL * c0, ORIGIN_X + PIXEL * c1
        y0, y1 = ORIGIN_Y - PIXEL * r0, ORIGIN_Y - PIXEL * r1
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        features.append(
            {
                'type': 'Feature',
                'properties': {'sector_id': name},
                'geometry': {'type': 'Polygon', 'coordinates': [ring]},
            }
        )
    collection = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': f'urn:ogc:def:crs:{CRS.replace(":", "::")}'},
        },
        'features': features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection, indent=2), encoding='utf-8')
    return path


def write_showcase_site(
    out_dir: Path,
    *,
    base_config: Path | None = None,
    **simulation_options: Any,
) -> dict[str, Any]:
    """Write the showcase site: MAP table, masks, zones, sectors, config, README.

    Layout::

        <out>/results/forecasts/slope_trf/residuals.npz|.json
        <out>/static/tsf_mask.tif
        <out>/static/sectors.geojson
        <out>/static/demo_zones.tif         (0 outside, 1 stable, 2 creep, 3 failure)
        <out>/static/roi_failure_mask.tif   (ground truth; compare --truth only)
        <out>/ale_showcase.yaml
        <out>/scenario.json
        <out>/README.txt

    The engines read only the table, the TSF mask and the sectors; the zones
    and the ground truth are for the presenter and the comparison report.
    """
    from ale_3levels.config import DEFAULT_CONFIG

    out_dir = Path(out_dir)
    simulation = simulate_showcase(**simulation_options)
    layout = simulation['layout']
    table = write_showcase_table(
        out_dir / 'results' / 'forecasts' / 'slope_trf',
        simulation,
        model_version='slope_trf@showcase',
    )
    tsf_mask = _write_raster(out_dir / 'static' / 'tsf_mask.tif', layout['mask'])
    zones = _write_raster(out_dir / 'static' / 'demo_zones.tif', layout['zones'])
    truth_mask = _write_raster(
        out_dir / 'static' / 'roi_failure_mask.tif', layout['failure']
    )
    sectors = _write_sectors(out_dir / 'static' / 'sectors.geojson', layout['sectors'])
    config = out_dir / 'ale_showcase.yaml'
    base = Path(base_config or DEFAULT_CONFIG).resolve()
    config.write_text(
        '# ALE configuration for the simulated showcase site (ale_3levels.demo).\n'
        '# Everything else (gates, levels, coherence) is the default configuration.\n'
        f'base: {base}\n'
        'input:\n'
        '  residuals: results/forecasts/slope_trf/residuals.npz\n'
        '  mask_file: static/tsf_mask.tif\n'
        '  model_id: trf\n'
        'site:\n'
        '  site_id: ale_showcase\n'
        'spatial:\n'
        '  sectors_file: static/sectors.geojson\n'
        '  sector_id_field: sector_id\n',
        encoding='utf-8',
    )
    dates = simulation['dates']
    counts = {
        name: int(np.count_nonzero(layout['zones'] == code))
        for name, code in SHOWCASE_ZONES.items()
        if code
    }
    scenario = {
        'grid': list(layout['mask'].shape),
        'pixel_m': PIXEL,
        'acquisitions': len(dates),
        'calibration': [dates[0], dates[simulation['monitoring_index'] - 1]],
        'monitoring': [dates[simulation['monitoring_index']], dates[-1]],
        'failure_onset': simulation['failure_onset_date'],
        'atmospheric_artefact': simulation['aps_date'],
        'zone_codes': SHOWCASE_ZONES,
        'zone_pixels': counts,
        'sectors': list(layout['sectors']),
        'options': {k: v for k, v in simulation_options.items()},
    }
    (out_dir / 'scenario.json').write_text(
        json.dumps(scenario, indent=2), encoding='utf-8'
    )
    readme = out_dir / 'README.txt'
    readme.write_text(
        'Simulated ALE showcase site (ale_3levels.demo; not a MAP run).\n\n'
        f'TSF on a {layout["mask"].shape[0]}x{layout["mask"].shape[1]} grid of '
        f'{PIXEL:g} m pixels; {len(dates)} acquisitions every 12 days.\n'
        f'Calibration {scenario["calibration"][0]} .. {scenario["calibration"][1]}; '
        f'monitoring {scenario["monitoring"][0]} .. {scenario["monitoring"][1]}.\n\n'
        'Zones (static/demo_zones.tif):\n'
        '  stable   slow uniform settlement, forecast by MAP\n'
        '  creep    steady consolidation (~7.5 mm per 30 days), forecast by MAP\n'
        f'  failure  dam wall accelerating from {scenario["failure_onset"]}, '
        'not forecast by MAP\n'
        f'Atmospheric artefact on {scenario["atmospheric_artefact"]} (one date).\n\n'
        'Run: python3 subsystems/ale/scripts/run_demo.py (see subsystems/ale/DEMO.md).\n',
        encoding='utf-8',
    )
    return {
        'table': table,
        'tsf_mask': tsf_mask,
        'zones': zones,
        'truth_mask': truth_mask,
        'sectors': sectors,
        'config': config,
        'scenario': out_dir / 'scenario.json',
        'readme': readme,
    }
