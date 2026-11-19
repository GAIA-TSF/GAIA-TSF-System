"""Test helpers: a small MAP-style residual table (schema 2.0).

Not named conftest.py: MAP tests import their own ``conftest`` module by name,
and a second conftest.py would shadow it when both suites are collected.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HEIGHT, WIDTH = 3, 4
# Irregular cadence: 12-day revisits with one 24-day gap.
OFFSETS = (0, 12, 24, 36, 60, 72, 84, 96, 108, 120)
DATES = tuple((date(2020, 1, 1) + timedelta(days=d)).isoformat() for d in OFFSETS)
MONITORING_START = DATES[6]


def cumulative_m(index: int, row: int, col: int) -> float:
    """Synthetic cumulative LOS in metres (negative = away from the satellite)."""
    return -0.001 * OFFSETS[index] / 12.0 * (1 + row) + 0.0002 * col


def write_map_table(
    directory: Path,
    *,
    leads: tuple[int, ...] = (1, 2),
    target_units: str = 'mm',
    sidecar_updates: dict | None = None,
    columns_update=None,
) -> Path:
    """Write ``residuals.npz`` + ``residuals.json`` in MAP schema 2.0 form."""
    columns: dict[str, list] = {
        name: []
        for name in (
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
    }
    for origin in range(len(DATES)):
        for lead in leads:
            target = origin + lead
            expected = target >= len(DATES)
            for row in range(HEIGHT):
                for col in range(WIDTH):
                    persistence = cumulative_m(origin, row, col)
                    predicted = persistence - 0.0009 * lead
                    observed = np.nan if expected else cumulative_m(target, row, col)
                    target_date = (
                        (
                            date.fromisoformat(DATES[-1])
                            + timedelta(days=12 * (target - 9))
                        ).isoformat()
                        if expected
                        else DATES[target]
                    )
                    in_monitoring = target_date >= MONITORING_START
                    role = 'none'
                    if not in_monitoring:
                        role = (
                            'train'
                            if target < 3
                            else 'validation'
                            if target < 4
                            else 'test'
                        )
                    columns['series_id'].append(row * WIDTH + col)
                    columns['row'].append(row)
                    columns['col'].append(col)
                    columns['x'].append(500005.0 + 10 * col)
                    columns['y'].append(6999995.0 - 10 * row)
                    columns['forecast_origin_date'].append(DATES[origin])
                    columns['lead'].append(lead)
                    columns['target_date'].append(target_date)
                    columns['target_date_status'].append(
                        'expected' if expected else 'acquired'
                    )
                    columns['last_observation_date'].append(DATES[origin])
                    columns['persistence_value'].append(persistence)
                    columns['predicted_value'].append(predicted)
                    columns['prediction_lower'].append(predicted - 0.001)
                    columns['prediction_upper'].append(predicted + 0.001)
                    columns['period'].append(
                        'monitoring' if in_monitoring else 'calibration'
                    )
                    columns['sample_role'].append(role)
                    columns['observed_value'].append(observed)
                    columns['residual'].append(observed - predicted)
    arrays = {name: np.asarray(values) for name, values in columns.items()}
    if columns_update is not None:
        columns_update(arrays)
    sidecar = {
        'schema_version': '2.0',
        'model': 'trf',
        'model_version': 'slope_trf@test',
        'target_variable': 'cumulative_los',
        'target_feature': 'cumulative_los_current',
        'target_units': target_units,
        'horizon': max(leads),
        'calibration_window': [DATES[0], DATES[5]],
        'monitoring_window': [MONITORING_START, DATES[-1]],
        'grid': {
            'crs': 'EPSG:32735',
            'transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0],
            'height': HEIGHT,
            'width': WIDTH,
        },
        'product': 'map_forecast_residuals',
        **(sidecar_updates or {}),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'residuals.npz'
    np.savez_compressed(path, **arrays)
    path.with_suffix('.json').write_text(json.dumps(sidecar), encoding='utf-8')
    return path


def site_table(tmp_path: Path) -> Path:
    """A valid table under ``<tmp>/site_a/results/forecasts/slope_trf``."""
    return write_map_table(tmp_path / 'site_a' / 'results' / 'forecasts' / 'slope_trf')
