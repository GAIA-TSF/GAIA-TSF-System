"""B2.1: unified z = (e - mu_cal) / sigma_cal, frozen before monitoring."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ale_3levels.config import load_config
from ale_3levels.contract import load_contract
from ale_3levels.engines.three_level.level2_cusum import aggregate_residual
from ale_3levels.engines.three_level.pipeline import (
    ThreeLevelPipeline,
    inputs_from_contract,
)
from ale_fixtures import write_long_table


def _config(formula):
    config = load_config()['engine_configs']['three_level']
    return {**config, 'zscore_formula': formula}


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    table = write_long_table(tmp_path_factory.mktemp('table'))
    return inputs_from_contract(load_contract(table))


def _calibration(run, level):
    return json.loads(open(run.metadata[f'{level}_calibration_path']).read())


def test_l1_z_is_centred_on_the_frozen_calibration_mean(inputs, tmp_path):
    run = ThreeLevelPipeline(_config('unified'), tmp_path).run(inputs)
    l1 = _calibration(run, 'l1')
    mask = inputs['mask']
    expected = (inputs['residual_stack'] - l1['mean_residual_diagnostic']) / l1['sigma']
    np.testing.assert_array_equal(run.arrays['residual_z'][:, mask], expected[:, mask])
    assert run.metadata['level1']['formula'] == 'z = (e - mu_cal) / sigma_cal'
    assert run.metadata['level1']['center'] == l1['mean_residual_diagnostic']


def test_l2_z_is_centred_on_its_own_calibration_mean(inputs, tmp_path):
    run = ThreeLevelPipeline(_config('unified'), tmp_path).run(inputs)
    l2 = _calibration(run, 'l2')
    series = aggregate_residual(inputs['residual_stack'], inputs['mask'])
    expected = (series - l2['mean_residual']) / l2['sigma']
    end = inputs['monitoring_window'][1]
    np.testing.assert_array_equal(run.details['level2'].z[:end], expected[:end])


def test_frozen_sigma_keeps_the_prototype_formula(inputs, tmp_path):
    run = ThreeLevelPipeline(_config('frozen_sigma'), tmp_path).run(inputs)
    l1 = _calibration(run, 'l1')
    mask = inputs['mask']
    expected = inputs['residual_stack'] / l1['sigma']
    np.testing.assert_array_equal(run.arrays['residual_z'][:, mask], expected[:, mask])
    assert 'center' not in run.metadata['level1']


def test_parameters_are_frozen_before_monitoring(inputs, tmp_path):
    changed = dict(inputs)
    start = inputs['monitoring_window'][0]
    residuals = inputs['residual_stack'].copy()
    residuals[start + 5 :] *= 10.0
    changed['residual_stack'] = residuals
    a = ThreeLevelPipeline(_config('unified'), tmp_path / 'a').run(inputs)
    b = ThreeLevelPipeline(_config('unified'), tmp_path / 'b').run(changed)
    for level in ('l1', 'l2'):
        ca, cb = _calibration(a, level), _calibration(b, level)
        assert (
            ca['sigma'],
            ca.get('mean_residual_diagnostic', ca.get('mean_residual')),
        ) == (
            cb['sigma'],
            cb.get('mean_residual_diagnostic', cb.get('mean_residual')),
        )
    np.testing.assert_array_equal(
        a.arrays['residual_z'][: start + 5], b.arrays['residual_z'][: start + 5]
    )
