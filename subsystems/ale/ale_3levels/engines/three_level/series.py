"""Per-acquisition monitoring series with contract v1.0 canonical names.

``monitoring_series.json``/``.csv`` are the main numerical output of the
monitoring pipeline; the animation graph export reuses the same field names.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ale_3levels.engines.three_level.calibration import CONTRACT_VERSION, SCHEMA_VERSION

RATE = 'rate'
ACCELERATION = 'acceleration'

# Canonical order and units. RATE/ACCELERATION resolve to the export's
# deformation-rate unit (native mm/day for the main output).
FIELDS: dict[str, str] = {
    'date': 'ISO date (target date)',
    'forecast_origin_date': 'ISO date',
    'lead_time_days': 'days',
    'lead_time_acquisitions': 'acquisitions',
    'phase': 'calibration | monitoring | outside',
    'quality_state': 'contract 7',
    'observed_tsf_coverage': 'fraction',
    'observed_mean_los_velocity': RATE,
    'predicted_baseline_velocity': RATE,
    'prediction_spread': RATE,
    'mean_residual_rate': RATE,
    'l1_anomalous_pixels': 'pixels',
    'l1_persistent_anomalous_pixels': 'pixels',
    'l1_coherent_anomalous_pixels': 'pixels',
    'l2_z': 'sigma',
    'l2_instability_cusum': 'sigma',
    'l2_overforecast_cusum': 'sigma',
    'l2_alarm': 'boolean',
    'coherent_region_available': 'boolean',
    'coherent_region_l2_instability_cusum': 'sigma',
    'coherent_region_l2_overforecast_cusum': 'sigma',
    'coherent_region_l2_alarm': 'boolean',
    'regime_change_score': 'dimensionless heuristic in [0, 1); not a probability',
    'coherent_region_regime_change_score': (
        'dimensionless heuristic in [0, 1); not a probability'
    ),
    'l3_level': 'normal | watch | alert | not_evaluated',
    'l3_sustained_watch_score': 'boolean',
    'l3_sustained_alert_score': 'boolean',
    'l3_l2_alarm': 'boolean',
    'l3_coherent_region': 'boolean',
    'acceleration_cusum_input': ACCELERATION,
    'acceleration_cusum_instability': 'dimensionless',
    'acceleration_cusum_recovery': 'dimensionless',
    'dynamics': 'accelerating | decelerating | stable',
    'oscillation': 'boolean',
    'coherent_region_acceleration_cusum_instability': 'dimensionless',
    'coherent_region_acceleration_cusum_recovery': 'dimensionless',
    'coherent_region_dynamics': 'accelerating | decelerating | stable',
}


def field_units(names: Sequence[str], rate_unit: str) -> dict[str, str]:
    """Units of the given canonical fields for one export."""
    resolved = {RATE: rate_unit, ACCELERATION: f'{rate_unit}/day' if rate_unit else ''}
    return {name: resolved.get(FIELDS[name], FIELDS[name]) for name in names}


def build_records(
    columns: Mapping[str, Sequence[Any] | np.ndarray],
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Rows in canonical field order; NaN becomes ``None`` (JSON null)."""
    unknown = sorted(set(columns) - set(FIELDS))
    if unknown:
        raise KeyError(f'Non-canonical monitoring series fields: {unknown}')
    names = [name for name in FIELDS if name in columns]
    return [{name: _value(columns[name][index]) for name in names} for index in indices]


def write_series(
    path: Path,
    records: list[dict[str, Any]],
    *,
    rate_unit: str,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write ``<path>.json`` (with units and metadata) and ``<path>.csv``."""
    names = list(records[0]) if records else []
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix('.json')
    csv_path = path.with_suffix('.csv')
    payload = {
        'schema_version': SCHEMA_VERSION,
        'contract_version': CONTRACT_VERSION,
        'product': 'map_tsf_monitoring_series',
        **metadata,
        'units': field_units(names, rate_unit),
        'series': records,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8'
    )
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(
            {key: '' if value is None else value for key, value in row.items()}
            for row in records
        )
    return json_path, csv_path


def phases(
    size: int, calibration_window: tuple[int, int], monitoring_window: tuple[int, int]
) -> np.ndarray:
    """Monitoring wins where the configured windows overlap."""
    output = np.full(size, 'outside', dtype='<U11')
    output[calibration_window[0] : calibration_window[1]] = 'calibration'
    output[monitoring_window[0] : monitoring_window[1]] = 'monitoring'
    return output


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    text = str(value)
    return text or None
