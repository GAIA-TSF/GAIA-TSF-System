"""Predictors must be known at the forecast origin (no future data)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from subsystems.map.dataset.dataset_builder import validate_predictors

MAP_CONFIG = Path(__file__).resolve().parents[1] / 'config.yaml'
DATASETS = yaml.safe_load(MAP_CONFIG.read_text(encoding='utf-8'))['datasets']


@pytest.mark.parametrize(
    'name',
    [
        'velocity_lag1',
        'acceleration_lag3',
        'velocity_current',
        'velocity_roll_mean',
        'velocity_roll_std',
        'displacement_increment_lag1',
        'lag1',
        'amd_diff1',
        'roll_mean',
        'precip_30d',
        'temp_7d_mean',
        'annual_sin',
        'annual_cos',
    ],
)
def test_features_known_at_the_origin_are_allowed(name: str) -> None:
    validate_predictors([name])


@pytest.mark.parametrize(
    'name',
    ['velocity_smooth', 'velocity_smooth_lag1', 'velocity_lead1', 'amd_next', 'x_future'],
)
def test_features_using_later_acquisitions_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        validate_predictors(['velocity_lag2', name])


@pytest.mark.parametrize('dataset', sorted(DATASETS))
def test_configured_datasets_pass(dataset: str) -> None:
    validate_predictors(DATASETS[dataset]['features'])
