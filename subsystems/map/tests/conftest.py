"""Minimal dated DAG-style fixture projects for both MAP key variables."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from subsystems.dag.plugins.features.amd_temporal_features import compute_amd_temporal
from subsystems.map.utils.config_loader import load_config


MAP_CONFIG = Path(__file__).resolve().parents[1] / 'config.yaml'
HEIGHT, WIDTH, TIMES = 4, 5, 40
DATES = tuple((date(2020, 1, 1) + timedelta(days=12 * i)).isoformat() for i in range(TIMES))
TRANSFORM = from_origin(500000, 7000000, 10, 10)


def write_stack(path: Path, values: np.ndarray, dates: tuple[str, ...] = DATES) -> None:
    """Write a float32 GeoTIFF stack with acquisition dates as descriptions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype='float32')
    if values.ndim == 2:
        values = values[np.newaxis]
    with rasterio.open(
        path, 'w', driver='GTiff', width=WIDTH, height=HEIGHT, count=values.shape[0],
        dtype='float32', crs='EPSG:32633', transform=TRANSFORM, nodata=np.nan,
    ) as target:
        target.write(values)
        if values.shape[0] == len(dates):
            target.descriptions = dates


def lag(stack: np.ndarray, order: int) -> np.ndarray:
    """DAG slope lag: row t holds the value from acquisition t - order."""
    output = np.full(stack.shape, np.nan)
    output[order:] = stack[:-order]
    return output


def slope_stacks(seed: int = 0) -> dict[str, np.ndarray]:
    """Cumulative LOS in mm moving away from the satellite, with gaps."""
    rng = np.random.default_rng(seed)
    days = np.arange(TIMES)[:, None, None] * 12.0
    rate = -0.05 - 0.05 * rng.random((1, HEIGHT, WIDTH))
    cumulative = rate * days + 0.5 * rng.normal(size=(TIMES, HEIGHT, WIDTH))
    cumulative[17, 1, 1] = np.nan  # isolated missing observation
    velocity = np.full(cumulative.shape, np.nan)
    velocity[1:] = np.diff(cumulative, axis=0) / 12.0
    acceleration = np.full(cumulative.shape, np.nan)
    acceleration[1:] = np.diff(velocity, axis=0) / 12.0
    stacks = {'cumulative_los_current': cumulative}
    for order in (1, 2, 3):
        stacks[f'velocity_lag{order}'] = lag(velocity, order)
        stacks[f'acceleration_lag{order}'] = lag(acceleration, order)
    return stacks


def amd_stacks(seed: int = 1) -> dict[str, np.ndarray]:
    """AMD band difference with cloud gaps; predictors from DAG's own code."""
    rng = np.random.default_rng(seed)
    t = np.arange(TIMES)[:, None, None]
    base = 0.02 * np.sin(2 * np.pi * t / 30) + 0.01 * rng.random((1, HEIGHT, WIDTH))
    data = (base + 0.002 * rng.normal(size=(TIMES, HEIGHT, WIDTH))).astype(np.float32)
    data[rng.random(data.shape) < 0.15] = np.nan  # cloud-masked pixels
    data[22, 2, 3] = np.nan
    days = tuple(date.fromisoformat(d) for d in DATES)
    features = compute_amd_temporal(data, days, {'rolling_window': 3, 'min_periods': 1})
    return {'amd_difference': data, **features}


def write_project(root: Path, kv: str, stacks: dict[str, np.ndarray] | None = None) -> Path:
    """Write ``static`` masks and DAG feature stacks below ``root``."""
    stacks = stacks or (slope_stacks() if kv == 'slope' else amd_stacks())
    mask = np.ones((HEIGHT, WIDTH))
    mask[0, 0] = 0
    stable = np.zeros((HEIGHT, WIDTH))
    stable[2:, :] = 1
    names = ('tsf_mask.tif', 'tsf_stable.tif') if kv == 'slope' else ('amd_mask.tif', None)
    write_stack(root / 'static' / names[0], mask, ())
    if names[1]:
        write_stack(root / 'static' / names[1], stable, ())
    directory = root / 'results' / ('temporal_features' if kv == 'slope' else 'features')
    for name, values in stacks.items():
        target_dir = directory if kv == 'slope' or name == 'amd_difference' else root / 'results' / 'temporal_features'
        write_stack(target_dir / f'{name}.tif', values)
    for folder in {directory, root / 'results' / 'temporal_features'}:
        (folder / 'metadata.json').write_text(json.dumps({'acquisition_dates': list(DATES)}))
    return root


def project_config(root: Path, dataset: str, model: str, horizon: int = 3) -> dict:
    """The shipped MAP config with fixture paths, dates, and small models."""
    config = deepcopy(load_config(MAP_CONFIG, dataset, model))
    entry = config['datasets'][dataset]
    entry['experiment_dir'] = str(root)
    entry['temporal_windows'] = {
        'calibration': {'start_date': DATES[0], 'end_date': DATES[28]},
        'monitoring': {'start_date': DATES[28], 'end_date': DATES[-1]},
    }
    config['forecast']['horizon'] = horizon
    config['models']['trf'].update(n_estimators=10, n_jobs=1)
    config['models']['lstm'].update(hidden_size=4, num_layers=1, look_back=3)
    config['models']['lstm']['training'].update(epochs=2, batch_size=64)
    return config


@pytest.fixture
def slope_project(tmp_path: Path) -> Path:
    return write_project(tmp_path / 'slope', 'slope')


@pytest.fixture
def amd_project(tmp_path: Path) -> Path:
    return write_project(tmp_path / 'amd', 'amd')
