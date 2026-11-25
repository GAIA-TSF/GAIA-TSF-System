"""Common output schema and writers (identical for both engines).

``<out>/<engine>/`` contains:

* ``alerts.npz``: per pixel and acquisition (``time, row, col``): ``level``
  (codes into ``levels``), ``severity`` (0-3, -1 NO_DATA), ``trigger_reason``
  (codes into ``trigger_reasons``), ``level_scope`` (pixel/region),
  ``region_id`` and ``metric_<name>`` stacks; plus ``dates`` and the TSF
  ``mask``. One raster per acquisition per field.
* ``alerts.json``: engine, engine_version, config_hash, vocabularies,
  severity mapping, recommended actions, engine attributes and the MAP
  input identity (model_version, tsf_version, table SHA-256, ...).
* ``history.csv``: per acquisition, pixel counts per level and summaries of
  each metric over valid pixels (numerical time history).
* ``sectors.csv`` / ``sectors.json``: per sector and acquisition, the
  aggregated level and pixel counts (see :func:`ale_3levels.spatial.aggregate_sectors`).

Files are written deterministically (sorted keys, no timestamps, ``.npz``
entries with a fixed zip date), so the same input and configuration give
byte-identical outputs.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ale_3levels.contract import CALIBRATION, ForecastContract
from ale_3levels.engines.base import LEVEL_SCOPES, NO_DATA, SEVERITY, AlertResult
from ale_3levels.spatial import SectorSummary

SCHEMA_VERSION = '1.0'
# Legacy metric names (B2.4). The MAP prototype called the L3 score P_regime
# (earlier regime_risk), which suggested a probability; it is a heuristic
# score in [0, 1) and is written as regime_change_score.
METRIC_ALIASES = {
    'three_level': {
        'P_regime': 'regime_change_score',
        'regime_risk': 'regime_change_score',
    },
    'fixed_gates': {},
}
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SEVERITY_NOTE = (
    'Ordinal severity for comparison only: WHITE/NONE=0, GREEN/L1=1, '
    'YELLOW/L2=2, RED/L3=3, NO_DATA=-1. The levels of the two engines do not '
    'mean the same thing.'
)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """``np.savez_compressed`` equivalent with a fixed timestamp (reproducible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asanyarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f'{name}.npy', date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + '\n',
        encoding='utf-8',
    )
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def input_identity(contract: ForecastContract) -> dict[str, Any]:
    """MAP input identity recorded with every engine output."""
    attrs = contract.attrs
    keys = (
        'site_id',
        'model_id',
        'model_version',
        'tsf_version',
        'tsf_version_provisional',
        'table_path',
        'table_sha256',
        'table_schema_version',
        'contract_version',
        'source_units',
        'declared_units',
        'los_sign_convention',
        'reference_date',
        'reference_point',
        'monitoring_start',
        'lead',
    )
    identity = {key: attrs.get(key) for key in keys}
    identity['calibration_split_dates'] = int(
        np.any(contract.split == CALIBRATION, axis=(1, 2)).sum()
    )
    identity['dates'] = {
        'count': int(contract.dates.size),
        'first': str(contract.dates[0]),
        'last': str(contract.dates[-1]),
    }
    return identity


def write_engine_output(
    result: AlertResult,
    contract: ForecastContract,
    sectors: SectorSummary,
    out_dir: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write one engine's result in the common schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        'dates': np.asarray(result.dates, dtype='<U10'),
        'mask': np.asarray(contract.mask, dtype=bool),
        'level': result.level.astype(np.int8),
        'severity': result.severity.astype(np.int8),
        'trigger_reason': result.trigger_reason.astype(np.int16),
        'level_scope': result.level_scope.astype(np.int8),
        'region_id': result.region_id.astype(np.int32),
        **{
            f'metric_{name}': np.asarray(values, dtype=np.float64)
            for name, values in result.metric_values.items()
        },
    }
    paths = {'alerts': write_npz(out_dir / 'alerts.npz', arrays)}
    metadata = {
        'schema_version': SCHEMA_VERSION,
        'engine': result.engine,
        'engine_version': result.engine_version,
        'config_hash': result.config_hash,
        'levels': list(result.levels),
        'severity_mapping': {
            **{label: value for label, value in SEVERITY[result.engine].items()},
            NO_DATA: -1,
        },
        'severity_note': SEVERITY_NOTE,
        'trigger_reasons': list(result.trigger_reasons),
        'level_scopes': list(LEVEL_SCOPES),
        'metrics': sorted(result.metric_values),
        'metric_aliases': {
            alias: target
            for alias, target in METRIC_ALIASES[result.engine].items()
            if target in result.metric_values
        },
        'recommended_actions': result.recommended_actions,
        'recommended_actions_note': 'Advisory text only; ALE executes no action.',
        'engine_attrs': result.attrs,
        'input': input_identity(contract),
        'grid': {
            'crs': contract.grid.crs,
            'transform': list(contract.grid.transform),
            'height': contract.grid.height,
            'width': contract.grid.width,
        },
        'sector_rule': sectors.rule,
        **(extra or {}),
    }
    paths['metadata'] = write_json(out_dir / 'alerts.json', metadata)
    paths['history'] = _write_history(out_dir / 'history.csv', result, contract)
    paths['sectors'] = _write_sectors(out_dir / 'sectors.csv', result, sectors)
    paths['sectors_json'] = write_json(
        out_dir / 'sectors.json',
        {
            'engine': result.engine,
            'engine_version': result.engine_version,
            'config_hash': result.config_hash,
            'sector_ids': list(sectors.sector_ids),
            'rule': sectors.rule,
        },
    )
    return paths


def _fmt(value: float) -> str:
    return '' if not np.isfinite(value) else repr(float(value))


def _write_history(path: Path, result: AlertResult, contract: ForecastContract) -> Path:
    mask = contract.mask
    labels = list(result.levels)
    metrics = sorted(result.metric_values)
    header = [
        'date',
        'valid_pixels',
        *[f'pixels_{label}' for label in labels],
        'max_severity',
    ]
    for name in metrics:
        header += [f'{name}_median', f'{name}_p95', f'{name}_max']
    rows = []
    for index, day in enumerate(result.dates):
        level = result.level[index][mask]
        severity = result.severity[index][mask]
        valid = severity >= 0
        row = [
            str(day),
            int(valid.sum()),
            *[int(np.count_nonzero(level == code)) for code in range(len(labels))],
            int(severity.max()) if severity.size else -1,
        ]
        for name in metrics:
            values = result.metric_values[name][index][mask]
            values = values[np.isfinite(values)]
            if values.size:
                row += [
                    _fmt(np.median(values)),
                    _fmt(np.percentile(values, 95)),
                    _fmt(values.max()),
                ]
            else:
                row += ['', '', '']
        rows.append(row)
    return _write_csv(path, header, rows)


def _write_sectors(path: Path, result: AlertResult, sectors: SectorSummary) -> Path:
    native = {value: label for label, value in SEVERITY[result.engine].items()}
    native[-1] = NO_DATA
    header = [
        'date',
        'sector_id',
        'level',
        'severity',
        'valid_pixels',
        'pixels_no_data',
        'pixels_severity_0',
        'pixels_severity_1',
        'pixels_severity_2',
        'pixels_severity_3',
        'coherent_pixels_ge_1',
        'coherent_pixels_ge_2',
        'coherent_pixels_ge_3',
    ]
    rows = []
    for index, day in enumerate(result.dates):
        for position, sector in enumerate(sectors.sector_ids):
            severity = int(sectors.severity[index, position])
            rows.append(
                [
                    str(day),
                    sector,
                    native[severity],
                    severity,
                    int(sectors.valid_pixels[index, position]),
                    *[int(v) for v in sectors.pixel_counts[index, position]],
                    *[int(v) for v in sectors.coherent_counts[index, position]],
                ]
            )
    return _write_csv(path, header, rows)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)
    return path


def read_alerts(engine_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read ``alerts.npz`` and ``alerts.json`` of one engine output directory."""
    with np.load(engine_dir / 'alerts.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads((engine_dir / 'alerts.json').read_text(encoding='utf-8'))
    return arrays, metadata


def get_metric(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], name: str
) -> np.ndarray:
    """A metric stack by name or legacy alias (e.g. ``P_regime``)."""
    target = metadata.get('metric_aliases', {}).get(name, name)
    key = f'metric_{target}'
    if key not in arrays:
        raise KeyError(f'{metadata.get("engine")} output has no metric {name!r}.')
    return arrays[key]
