"""Explicit forecast valid dates and compatibility checks for MAP artifacts."""

import json
from datetime import date
from pathlib import Path
from typing import Any

from subsystems.map.utils.contract import SCHEMA_VERSION


def forecast_timing(dataset: Any) -> dict[str, Any]:
    """Describe origin as latest predictor observation, not model fit/issue time.

    Lead time (contract 3.2) is ``target_date - forecast_origin_date`` in days
    and in acquisitions of the dataset date axis.
    """
    origins = dataset.forecast_origin_dates or (None,) * len(dataset.dates)
    sources = dataset.source_dates or dataset.dates
    index = {value: position for position, value in enumerate(dataset.dates)}
    series = []
    for position, (source, origin, target) in enumerate(
        zip(sources, origins, dataset.dates)
    ):
        days = (
            None
            if origin is None
            else (date.fromisoformat(target) - date.fromisoformat(origin)).days
        )
        acquisitions = (
            position - index[origin] if origin is not None and origin in index else None
        )
        series.append(
            {
                'source_label_date': source,
                'forecast_origin_date': origin,
                'target_date': target,
                'lead_time_days': days,
                'lead_time_acquisitions': acquisitions,
                'timing_state': (
                    'unknown'
                    if origin is None
                    else 'forecast'
                    if origin < target
                    else 'reconstruction'
                ),
            }
        )
    return {
        'schema_version': SCHEMA_VERSION,
        'target_feature': dataset.target_feature,
        'date_axis': 'target_date',
        'forecast_origin_definition': 'latest_predictor_observation_date',
        'lead_time': lead_time_summary(series),
        'series': series,
    }


def lead_time_summary(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimum/maximum lead over rows that have a forecast origin."""

    def bounds(key: str) -> dict[str, int | None]:
        values = [row[key] for row in series if row.get(key) is not None]
        return {
            'min': min(values) if values else None,
            'max': max(values) if values else None,
        }

    return {
        'acquisitions': bounds('lead_time_acquisitions'),
        'days': bounds('lead_time_days'),
    }


def validate_model_target(model_path: Path, target: str, features: list[str]) -> None:
    """Prevent silently reusing a lag-target model for a current target."""
    metadata_path = model_path.parent / 'experiment.json'
    if not metadata_path.is_file():
        raise ValueError(
            'Model target identity is unavailable; run learning to create '
            f'experiment.json before inference: {model_path}'
        )
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if (
        metadata.get('target_feature') != target
        or metadata.get('feature_names') != features
    ):
        raise ValueError(
            'Configured target or ordered features differ from the trained model. '
            'Run learning for this configuration before inference.'
        )
