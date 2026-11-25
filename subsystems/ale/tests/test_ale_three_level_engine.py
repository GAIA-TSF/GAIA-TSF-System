"""Engine B adapter: common schema without changing engine-B numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ale_3levels.config import load_config
from ale_3levels.contract import MONITORING, ForecastContract, Grid
from ale_3levels.engines.base import LEVEL_SCOPES, NO_REGION, Engine
from ale_3levels.engines.three_level.engine import (
    LEVELS,
    TRIGGER_REASONS,
    ThreeLevelEngine,
    to_alert_result,
)
from ale_3levels.engines.three_level.pipeline import ThreeLevelPipeline, inputs_from_npz

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'synthetic_crop'


def _contract_from_inputs(inputs) -> ForecastContract:
    """A contract carrying exactly the golden prototype inputs."""
    observed = inputs['observed_stack']
    shape = observed.shape
    start = inputs['monitoring_window'][0]
    zeros = np.zeros(shape)
    return ForecastContract(
        attrs={
            'target_feature': 'cumulative_los_current',
            'monitoring_start': inputs['dates'][start],
        },
        grid=Grid(inputs['crs'], inputs['transform'], *shape[1:]),
        lead=1,
        dates=np.asarray(inputs['dates']),
        forecast_origin=np.asarray(inputs['forecast_origin_dates']),
        delta_t_days=np.full(shape, 12.0),
        observed_mm=zeros,
        predicted_mm=zeros,
        persistence_mm=zeros,
        residual_mm=zeros,
        observed_mm_per_day=observed,
        predicted_mm_per_day=inputs['prediction_stack'],
        residual_mm_per_day=inputs['residual_stack'],
        split=np.full(shape, MONITORING, dtype=np.int8),
        mask=inputs['mask'],
        valid_mask=np.broadcast_to(inputs['mask'], shape).copy(),
        series_dates=np.asarray(inputs['dates']),
        observed_series_mm=zeros,
    )


@pytest.fixture(scope='module')
def golden(tmp_path_factory):
    inputs = inputs_from_npz(GOLDEN / 'inputs.npz')
    config = load_config()['engine_configs']['three_level']
    # Pre-B2 prototype z-score and calibration: the golden inputs carry no split.
    config = {
        **config,
        'zscore_formula': 'frozen_sigma',
        'calibration': {**config['calibration'], 'population': 'prototype'},
    }
    run = ThreeLevelPipeline(config, tmp_path_factory.mktemp('b')).run(inputs)
    contract = _contract_from_inputs(inputs)
    result = to_alert_result(
        run, contract, engine_version='test', config_hash='x', recommended_actions={}
    )
    return inputs, run, result


def test_engine_b_numbers_pass_through_unchanged(golden):
    inputs, run, result = golden
    mask = inputs['mask']
    level2, level3 = run.details['level2'], run.details['level3']
    np.testing.assert_array_equal(result.metric_values['z'], run.arrays['residual_z'])
    for name, series in (
        ('cusum_pos', level2.instability_cusum),
        ('cusum_neg', level2.overforecast_cusum),
        ('regime_change_score', level3.evidence_score),
    ):
        expected = np.broadcast_to(series[:, None], (series.size, int(mask.sum())))
        np.testing.assert_array_equal(result.metric_values[name][:, mask], expected)
        assert np.all(np.isnan(result.metric_values[name][:, ~mask]))


def test_pixel_cusum_series_reproduces_the_prototype_maximum(golden):
    _, run, result = golden
    start, end = run.details['monitoring_window']
    series_max = np.fmax.reduce(
        result.metric_values['pixel_cusum_pos'][start:end], axis=0
    )
    prototype = run.arrays['level2_maximum_instability_cusum']
    valid = np.isfinite(prototype)
    np.testing.assert_array_equal(np.maximum(series_max, 0.0)[valid], prototype[valid])


def test_levels_follow_the_documented_rule(golden):
    _, run, result = golden
    start = run.details['monitoring_window'][0]
    labels = result.labels()
    mask = golden[0]['mask']
    assert np.all(labels[:start] == 'NO_DATA')
    assert np.all(labels[:, ~mask] == 'NO_DATA')
    level3 = run.details['level3']
    alert_dates = np.flatnonzero(level3.level == 'alert')
    assert alert_dates.size > 0
    regions = run.details['coherence'].regions
    assert regions
    for t in alert_dates:
        active = np.zeros_like(mask)
        for region in regions:
            if region.activation_index <= t:
                active |= region.support
        assert active.any()
        assert np.all(labels[t][mask & active] == 'L3')
        assert np.all(result.region_id[t][mask & active] >= 1)
        assert not np.any(labels[t][mask & ~active] == 'L3')
    not_alert = np.setdiff1d(np.arange(labels.shape[0]), alert_dates)
    assert not np.any(labels[not_alert] == 'L3')
    first_alert = run.metadata['level3']['first_alert_date']
    assert result.attrs['first_l3_alert_date'] == first_alert
    l1_only = labels == 'L1'
    assert np.all(run.arrays['anomaly_l1_binary'][l1_only])
    assert np.all(result.level_scope[l1_only] == LEVEL_SCOPES.index('pixel'))


def test_scope_and_region_are_consistent(golden):
    _, _, result = golden
    region = result.level_scope == LEVEL_SCOPES.index('region')
    assert np.all(result.region_id[region] >= 0)
    assert np.all(result.region_id[~region] == NO_REGION)
    assert set(np.unique(result.level)) <= set(range(len(LEVELS)))
    assert set(np.unique(result.trigger_reason)) <= set(range(len(TRIGGER_REASONS)))


def test_engine_satisfies_the_interface(tmp_path):
    engine = ThreeLevelEngine(load_config()['engine_configs']['three_level'], tmp_path)
    assert isinstance(engine, Engine)
    assert engine.name == 'three_level'
