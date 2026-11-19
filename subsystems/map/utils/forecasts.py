"""Forecast tables, uncertainty, error metrics, and observation matching.

Tables are long format: one row per (series, forecast origin, lead). They are
stored as compressed ``.npz`` columns with a ``.json`` sidecar holding the
schema, units, and run identity, the same formats MAP used before.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.dataset.dataset_builder import ForecastSamples


SCHEMA_VERSION = '2.0'
FORECAST_ORIGIN_DEFINITION = (
    'Acquisition date of the latest predictors; every predictor is known at the origin.'
)
LEAD_DEFINITION = 'Number of acquisitions after the forecast origin (1..horizon).'
RESIDUAL_DEFINITION = 'residual = observed_value - predicted_value, in target_units.'
# Units: "target" resolves to the KV's target_units in the JSON sidecar.
FIELDS: dict[str, str] = {
    'series_id': 'pixel index row * width + col on the dataset grid',
    'row': 'grid row',
    'col': 'grid column',
    'x': 'grid CRS, pixel centre',
    'y': 'grid CRS, pixel centre',
    'forecast_origin_date': 'ISO date',
    'lead': 'acquisitions',
    'target_date': 'ISO date',
    'target_date_status': 'acquired (on the loaded date axis) | expected (projected)',
    'last_observation_date': 'ISO date of the persistence value',
    'persistence_value': 'target',
    'predicted_value': 'target',
    'prediction_lower': 'target',
    'prediction_upper': 'target',
    'period': 'calibration | monitoring | outside (by target_date)',
    'sample_role': 'train | validation | test (reference normal data) | none',
    'observed_value': 'target',
    'residual': 'target',
}
FORECAST_FIELDS = tuple(name for name in FIELDS if name not in {'observed_value', 'residual'})


def target_dates(
    dates: tuple[str, ...],
    origin_indices: np.ndarray,
    horizon: int,
    interval_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(target_date, acquired, target_index)`` shaped ``(n, horizon)``.

    Targets on the loaded acquisition axis are ``acquired``. Leads beyond the
    last acquisition are ``expected`` at the configured nominal interval, or
    the median observed interval, after the last acquisition.
    """
    axis = np.asarray(dates, dtype='datetime64[D]')
    if interval_days is None:
        interval_days = float(np.median(np.diff(axis).astype(float))) if axis.size > 1 else 1.0
    index = origin_indices[:, None] + np.arange(1, horizon + 1)[None, :]
    acquired = index < axis.size
    beyond = np.maximum(index - (axis.size - 1), 0)
    projected = axis[-1] + np.round(beyond * interval_days).astype('timedelta64[D]')
    values = np.where(acquired, axis[np.minimum(index, axis.size - 1)], projected)
    return values.astype(str), acquired, index


def interval_offsets(
    observed: np.ndarray, predicted: np.ndarray, level: float
) -> dict[str, list[float | None] | float]:
    """Empirical per-lead residual quantiles from held-out validation forecasts."""
    residual = observed - predicted
    lower, upper, count = [], [], []
    for lead in range(residual.shape[1]):
        values = residual[np.isfinite(residual[:, lead]), lead]
        count.append(int(values.size))
        if values.size == 0:
            lower.append(None)
            upper.append(None)
            continue
        lower.append(float(np.quantile(values, (1.0 - level) / 2.0)))
        upper.append(float(np.quantile(values, (1.0 + level) / 2.0)))
    return {
        'method': 'empirical quantiles of validation residuals per lead',
        'level': level,
        'residual_lower': lower,
        'residual_upper': upper,
        'sample_count': count,
    }


def apply_interval(predicted: np.ndarray, uncertainty: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Add the per-lead residual quantiles to absolute predictions."""
    lower = np.asarray([np.nan if v is None else v for v in uncertainty['residual_lower']])
    upper = np.asarray([np.nan if v is None else v for v in uncertainty['residual_upper']])
    return predicted + lower[None, :], predicted + upper[None, :]


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    """MAE, RMSE, mean signed error (observed - predicted), valid count."""
    residual = (observed - predicted)[np.isfinite(observed) & np.isfinite(predicted)]
    if residual.size == 0:
        return {'mae': None, 'rmse': None, 'mean_signed_error': None, 'valid_count': 0}
    return {
        'mae': float(np.mean(np.abs(residual))),
        'rmse': float(np.sqrt(np.mean(np.square(residual)))),
        'mean_signed_error': float(np.mean(residual)),
        'valid_count': int(residual.size),
    }


def lead_metrics(
    samples: ForecastSamples,
    predicted: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Model and persistence errors per lead on identical valid observations."""
    rows = []
    persistence = np.broadcast_to(samples.anchors[:, None], predicted.shape)
    for lead in range(samples.horizon):
        observed = samples.targets[:, lead]
        model = error_metrics(observed, predicted[:, lead])
        baseline = error_metrics(observed, persistence[:, lead])
        row = {'lead': lead + 1, **model}
        row.update({f'constant_{key}': value for key, value in baseline.items() if key != 'valid_count'})
        row['mae_skill_vs_constant'] = (
            None
            if not model['valid_count'] or not baseline['mae']
            else 1.0 - model['mae'] / baseline['mae']
        )
        if lower is not None and upper is not None:
            valid = np.isfinite(observed) & np.isfinite(lower[:, lead])
            row['interval_coverage'] = (
                float(np.mean((observed[valid] >= lower[valid, lead]) & (observed[valid] <= upper[valid, lead])))
                if np.any(valid)
                else None
            )
        rows.append(row)
    return rows


def forecast_columns(
    samples: ForecastSamples,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    windows: dict[str, tuple[int, int]],
    interval_days: float | None = None,
) -> dict[str, np.ndarray]:
    """Long-format forecast rows, independent of observation availability."""
    horizon = samples.horizon
    count = samples.origin_indices.size
    dates = np.asarray(samples.dates)
    target_date, acquired, target_index = target_dates(
        samples.dates, samples.origin_indices, horizon, interval_days
    )
    width = samples.grid.width
    rows, cols = np.divmod(samples.pixel_indices, width)
    a, b, c, d, e, f = tuple(samples.grid.transform)[:6]
    x = a * (cols + 0.5) + b * (rows + 0.5) + c
    y = d * (cols + 0.5) + e * (rows + 0.5) + f
    period = np.full(target_index.shape, 'outside', dtype='<U11')
    for name in ('calibration', 'monitoring'):  # monitoring wins an overlap
        start, end = windows[name]
        period[(target_index >= start) & (target_index < end)] = name
    repeat = lambda values: np.repeat(values, horizon)  # noqa: E731
    return {
        'series_id': repeat(samples.pixel_indices),
        'row': repeat(rows),
        'col': repeat(cols),
        'x': repeat(x),
        'y': repeat(y),
        'forecast_origin_date': repeat(dates[samples.origin_indices]),
        'lead': np.tile(np.arange(1, horizon + 1), count),
        'target_date': target_date.ravel(),
        'target_date_status': np.where(acquired, 'acquired', 'expected').ravel(),
        'last_observation_date': repeat(dates[samples.anchor_indices]),
        'persistence_value': repeat(samples.anchors),
        'predicted_value': predicted.ravel(),
        'prediction_lower': lower.ravel(),
        'prediction_upper': upper.ravel(),
        'period': period.ravel(),
        'sample_role': np.full(count * horizon, 'none', dtype='<U10'),
    }


def match_observations(
    forecasts: dict[str, np.ndarray],
    observed_stack: np.ndarray,
    dates: tuple[str, ...],
    width: int,
) -> dict[str, np.ndarray]:
    """Attach observations by (series_id, target_date); keep origin and lead.

    Matching is by date, never by position, so a missing acquisition leaves
    only its own forecasts unmatched. Rows without a finite observation are
    dropped from the returned residual table.
    """
    position = {value: index for index, value in enumerate(dates)}
    time_index = np.asarray([position.get(value, -1) for value in forecasts['target_date']])
    rows, cols = np.divmod(forecasts['series_id'], width)
    observed = np.full(time_index.size, np.nan)
    on_axis = time_index >= 0
    observed[on_axis] = observed_stack[time_index[on_axis], rows[on_axis], cols[on_axis]]
    keep = np.isfinite(observed)
    matched = {name: values[keep] for name, values in forecasts.items()}
    matched['observed_value'] = observed[keep]
    matched['residual'] = matched['observed_value'] - matched['predicted_value']
    return matched


def row_keys(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Integer keys of (forecast origin, series, lead) for joining tables."""
    origin = columns['forecast_origin_date'].astype('datetime64[D]').astype(np.int64)
    series = columns['series_id'].astype(np.int64)
    return (origin << 40) + (series << 4) + columns['lead'].astype(np.int64)


def compare_residual_tables(
    tables: dict[str, dict[str, np.ndarray]],
    roles: tuple[str, ...] | None = None,
    period: str | None = None,
) -> list[dict[str, Any]]:
    """Compare models and the constant baseline on identical observations.

    Rows are intersected on (forecast origin, series, lead) across every table,
    optionally restricted to ``sample_role`` values or one ``period``.
    """
    selected = {}
    for name, table in tables.items():
        keep = np.ones(table['lead'].size, dtype=bool)
        if roles is not None:
            keep &= np.isin(table['sample_role'], roles)
        if period is not None:
            keep &= table['period'] == period
        selected[name] = {key: value[keep] for key, value in table.items()}
    common = None
    for table in selected.values():
        keys = row_keys(table)
        common = keys if common is None else np.intersect1d(common, keys)
    rows = []
    first = next(iter(selected.values()))
    for lead in np.unique(first['lead']):
        lead_keys = common[(common & 0xF) == lead]
        entry: dict[str, Any] = {'lead': int(lead)}
        for index, (name, table) in enumerate(selected.items()):
            keys = row_keys(table)
            order = np.argsort(keys)
            position = order[np.searchsorted(keys[order], lead_keys)]
            observed = table['observed_value'][position]
            if index == 0:
                entry['constant'] = error_metrics(observed, table['persistence_value'][position])
            entry[name] = error_metrics(observed, table['predicted_value'][position])
        rows.append(entry)
    return rows


def write_table(path: Path, columns: dict[str, np.ndarray], metadata: dict[str, Any]) -> Path:
    """Write ``<path>.npz`` columns and a ``<path>.json`` schema sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name in FIELDS if name in columns]
    np.savez_compressed(path.with_suffix('.npz'), **{name: columns[name] for name in names})
    units = {
        name: (metadata['target_units'] if FIELDS[name] == 'target' else FIELDS[name])
        for name in names
    }
    write_json(
        path.with_suffix('.json'),
        {
            'schema_version': SCHEMA_VERSION,
            'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'forecast_origin_definition': FORECAST_ORIGIN_DEFINITION,
            'lead_definition': LEAD_DEFINITION,
            'residual_definition': RESIDUAL_DEFINITION,
            'row_count': int(columns[names[0]].size) if names else 0,
            **metadata,
            'fields': units,
        },
    )
    return path.with_suffix('.npz')


def read_table(path: Path) -> dict[str, np.ndarray]:
    """Read a table written by :func:`write_table`."""
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON artifact, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')


def grid_metadata(samples: ForecastSamples) -> dict[str, Any]:
    """Spatial identity needed to rebuild rasters from series ids."""
    return {
        'grid': {
            'crs': str(samples.grid.crs),
            'transform': [float(v) for v in tuple(samples.grid.transform)[:6]],
            'height': int(samples.grid.height),
            'width': int(samples.grid.width),
        }
    }
