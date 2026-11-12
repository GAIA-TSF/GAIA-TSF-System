"""MAP monitoring contract v1.0: version constants and run-mode provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

CONTRACT_VERSION = '1.0'
SCHEMA_VERSION = '1.0'
# Artifacts written before the contract settled on "1.0" remain valid.
COMPATIBLE_SCHEMA_VERSIONS = frozenset({'1.0', '1.0.0'})

OPERATIONAL = 'operational'
RETROSPECTIVE = 'retrospective'
UNKNOWN = 'unknown'


def compatible_schema(value: Any) -> bool:
    """True for a persisted artifact written under contract v1.0."""
    return value in COMPATIBLE_SCHEMA_VERSIONS


def training_run_mode(
    experiment: dict[str, Any], monitoring_start_date: str
) -> tuple[str, str]:
    """Classify a trained model as operational or retrospective (contract 4.2).

    Operational: the training pixels come from the configured stable polygon
    (``configured_mask``), or every date used for training and pixel selection
    precedes monitoring. Anything else used monitoring-period information.
    """
    selection = experiment.get('calibration_spatial_selection') or {}
    method = selection.get('method')
    if method == 'configured_mask':
        return OPERATIONAL, 'training pixels from the configured stable polygon'
    ranges = experiment.get('split_date_ranges') or {}
    end_dates = [
        str(ranges[name]['end_date'])
        for name in ('train', 'validation')
        if isinstance(ranges.get(name), dict) and ranges[name].get('end_date')
    ]
    if selection.get('selection_end_date'):
        end_dates.append(str(selection['selection_end_date']))
    if method is None or not end_dates:
        return UNKNOWN, 'training provenance is missing from experiment.json'
    latest = max(end_dates)
    if latest < monitoring_start_date:
        return OPERATIONAL, f'training and pixel selection end {latest}, before monitoring'
    return (
        RETROSPECTIVE,
        f'{method} training or pixel selection uses dates up to {latest}, '
        f'after monitoring starts {monitoring_start_date}',
    )


def file_sha256(path: Path) -> str:
    """Content fingerprint of an input file such as the stable mask."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_training_inputs(experiment: dict[str, Any]) -> None:
    """Stop when the stable mask used for training has changed since."""
    selection = experiment.get('calibration_spatial_selection') or {}
    path, expected = selection.get('mask_file'), selection.get('mask_sha256')
    if not path or not expected:
        return
    if not Path(path).is_file():
        raise FileNotFoundError(f'Stable mask used for training is missing: {path}')
    if file_sha256(Path(path)) != expected:
        raise ValueError(
            f'Stable mask changed since training: {path}. Retrain the model.'
        )
