"""B2.2: engine-B baselines use only independent calibration records."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from ale_3levels.config import load_config
from ale_3levels.contract import CALIBRATION, TRAIN, VALIDATION, load_contract
from ale_3levels.engines.three_level.pipeline import (
    ThreeLevelPipeline,
    inputs_from_contract,
)
from ale_fixtures import write_long_table


def _config(population='independent', **calibration):
    config = load_config()['engine_configs']['three_level']
    return {
        **config,
        'calibration': {
            **config['calibration'],
            'population': population,
            **calibration,
        },
    }


@pytest.fixture(scope='module')
def contract(tmp_path_factory):
    table = write_long_table(tmp_path_factory.mktemp('table'))
    return load_contract(table)


def _calibration(run, level):
    return json.loads(open(run.metadata[f'{level}_calibration_path']).read())


def test_sigma_uses_only_calibration_split_records(contract, tmp_path):
    inputs = inputs_from_contract(contract)
    run = ThreeLevelPipeline(_config(), tmp_path / 'a').run(inputs)
    records = (contract.split == CALIBRATION) & contract.mask[None]
    expected = np.std(contract.residual_mm_per_day[records], ddof=1)
    l1 = _calibration(run, 'l1')
    assert l1['sigma'] == pytest.approx(expected, rel=1e-12)
    assert l1['sample_count'] == int(records.sum())
    assert l1['method'].endswith('+calibration_split')
    calibration_dates = contract.dates[np.any(records, axis=(1, 2))].tolist()
    assert l1['calibration_dates'] == calibration_dates
    assert _calibration(run, 'l2')['calibration_dates'] == calibration_dates


def test_training_and_validation_records_cannot_influence_baselines(contract, tmp_path):
    clean = inputs_from_contract(contract)
    corrupted = dict(clean)
    used_in_fitting = np.isin(contract.split, (TRAIN, VALIDATION))
    corrupted['residual_stack'] = np.where(
        used_in_fitting, clean['residual_stack'] * 1000.0, clean['residual_stack']
    )
    corrupted['observed_stack'] = np.where(
        used_in_fitting, clean['observed_stack'] * 1000.0, clean['observed_stack']
    )
    a = ThreeLevelPipeline(_config(), tmp_path / 'a').run(clean)
    b = ThreeLevelPipeline(_config(), tmp_path / 'b').run(corrupted)
    for level in ('l1', 'l2'):
        assert _calibration(a, level)['sigma'] == _calibration(b, level)['sigma']
    # The prototype population is affected by the same corruption.
    p = ThreeLevelPipeline(_config('prototype'), tmp_path / 'p').run(clean)
    q = ThreeLevelPipeline(_config('prototype'), tmp_path / 'q').run(corrupted)
    assert _calibration(p, 'l1')['sigma'] != _calibration(q, 'l1')['sigma']


def test_independent_sigma_differs_from_prototype(contract, tmp_path):
    inputs = inputs_from_contract(contract)
    independent = ThreeLevelPipeline(_config(), tmp_path / 'i').run(inputs)
    prototype = ThreeLevelPipeline(_config('prototype'), tmp_path / 'p').run(inputs)
    assert independent.metadata['calibration_population'] == 'independent'
    assert prototype.metadata['calibration_population'] == 'prototype'
    assert independent.metadata['n_calibration'] < prototype.metadata['n_calibration']


def test_n_calibration_is_recorded_and_short_calibration_warns(
    contract, tmp_path, caplog
):
    inputs = inputs_from_contract(contract)
    n_dates = int(
        np.any((contract.split == CALIBRATION) & contract.mask[None], axis=(1, 2)).sum()
    )
    with caplog.at_level(logging.WARNING):
        run = ThreeLevelPipeline(
            _config(minimum_dates=n_dates + 1), tmp_path / 'w'
        ).run(inputs)
    assert run.metadata['n_calibration'] == n_dates
    assert run.details['n_calibration'] == n_dates
    assert 'Independent calibration has' in caplog.text
    flags = [flag['flag'] for flag in run.metadata['quality']['run_quality_flags']]
    assert 'short_calibration' in flags


def test_independent_needs_calibration_records(contract, tmp_path):
    inputs = inputs_from_contract(contract)
    del inputs['calibration_records']
    with pytest.raises(ValueError, match='calibration_records'):
        ThreeLevelPipeline(_config(), tmp_path).run(inputs)


@pytest.mark.parametrize(
    ('key', 'value'), [('population', 'all'), ('mode', 'per_pixel')]
)
def test_invalid_calibration_settings(tmp_path, key, value):
    with pytest.raises(ValueError):
        ThreeLevelPipeline(_config(**{key: value}), tmp_path)
