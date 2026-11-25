"""B2.6: persisted CUSUM state, reset rules and continuation without look-ahead."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from ale_3levels.__main__ import main
from ale_3levels.config import load_config
from ale_3levels.contract import load_contract, read_mask
from ale_3levels.demo import write_demo_site
from ale_3levels.engines.three_level.level2_cusum import residual_cusum
from ale_3levels.engines.three_level.pipeline import (
    ThreeLevelPipeline,
    inputs_from_contract,
)
from ale_3levels.outputs import read_alerts
from ale_3levels.run import run
from ale_3levels.state import StateMismatchError, load_state

PREFIX, FULL = 50, 60


def test_residual_cusum_restarts_at_resets():
    z = np.array([-3.0, -3.0, -3.0, -3.0])
    plain, _ = residual_cusum(z, 0.5, 'negative', 0)
    reset, _ = residual_cusum(z, 0.5, 'negative', 0, frozenset({2}))
    assert plain.tolist() == [2.5, 5.0, 7.5, 10.0]
    assert reset.tolist() == [2.5, 5.0, 2.5, 5.0]


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp('state')
    full = write_demo_site(root / 'full', n_dates=FULL)
    prefix = write_demo_site(root / 'prefix', n_dates=PREFIX)
    return root, prefix, full


def _contract(site):
    return load_contract(site['table'], mask=read_mask(site['tsf_mask']))


def _config(**state):
    config = load_config()['engine_configs']['three_level']
    return {**config, 'state': {**config['state'], **state}}


def test_reset_after_l3_restarts_the_accumulators(demo, tmp_path):
    _, _, full = demo
    inputs = inputs_from_contract(_contract(full))
    plain = ThreeLevelPipeline(_config(), tmp_path / 'a').run(inputs)
    reset = ThreeLevelPipeline(_config(reset_after_l3=True), tmp_path / 'b').run(inputs)
    first_alert = int(np.flatnonzero(plain.details['level3'].level == 'alert')[0])
    resets = reset.details['resets']
    assert resets and resets[0] == {
        'date': inputs['dates'][first_alert + 1],
        'reason': 'l3_event',
    }
    index = first_alert + 1
    level2 = reset.details['level2']
    k = level2.parameters.reference_value
    assert level2.instability_cusum[index] == pytest.approx(
        max(0.0, -level2.z[index] - k)
    )
    assert (
        plain.details['level2'].instability_cusum[index]
        > level2.instability_cusum[index]
    )
    # Up to the first alert nothing changes.
    np.testing.assert_array_equal(
        plain.details['level2'].instability_cusum[: first_alert + 1],
        level2.instability_cusum[: first_alert + 1],
    )
    assert plain.details['resets'] == []


def test_state_round_trip(demo):
    root, prefix, _ = demo
    run('three_level', root / 'roundtrip', config_path=prefix['config'])
    state_dir = root / 'roundtrip' / 'three_level' / 'state'
    state = load_state(state_dir)
    assert state.identity == {
        'model_id': 'trf',
        'model_version': 'slope_trf@demo',
        'lead': 1,
    }
    assert set(state.calibration) == {'l1', 'l2', 'l3'}
    assert state.pixel_cusum_pos.shape == (24, 24)
    assert state.resets == ()


def test_continuing_from_state_equals_batch_without_look_ahead(demo):
    root, prefix, full = demo
    run('both', root / 'p', config_path=prefix['config'])
    run(
        'both',
        root / 'c',
        config_path=full['config'],
        state_dir=root / 'p' / 'three_level' / 'state',
    )
    run('both', root / 'b', config_path=full['config'])
    for engine in ('fixed_gates', 'three_level'):
        p, _ = read_alerts(root / 'p' / engine)
        c, meta = read_alerts(root / 'c' / engine)
        b, _ = read_alerts(root / 'b' / engine)
        n = p['level'].shape[0]
        for key in p:
            if key in ('dates', 'mask'):
                continue
            # Earlier issued levels and metrics are unchanged by later data.
            np.testing.assert_array_equal(c[key][:n], p[key], err_msg=f'{engine} {key}')
            # Continuing from the state equals a batch run.
            np.testing.assert_array_equal(c[key], b[key], err_msg=f'{engine} {key}')
    assert meta['engine_attrs']['resets'] == []
    assert meta['engine_attrs']['continued_from_state'] == p['dates'][-1]


def test_model_version_change_resets_at_the_first_new_date(demo):
    root, prefix, _ = demo
    run('three_level', root / 'v1', config_path=prefix['config'])
    other = write_demo_site(root / 'other_version', n_dates=FULL)
    sidecar = other['table'].with_suffix('.json')
    payload = json.loads(sidecar.read_text())
    payload['model_version'] = 'slope_trf@demo2'
    sidecar.write_text(json.dumps(payload))
    run(
        'three_level',
        root / 'v2',
        config_path=other['config'],
        state_dir=root / 'v1' / 'three_level' / 'state',
    )
    arrays, meta = read_alerts(root / 'v2' / 'three_level')
    previous_last = load_state(root / 'v1' / 'three_level' / 'state').last_date
    assert meta['engine_attrs']['resets'] == [
        {'date': _next_date(arrays, previous_last), 'reason': 'model_version_change'}
    ]


def test_recalibration_resets_at_the_first_new_date(demo):
    root, prefix, full = demo
    run('three_level', root / 'r1', config_path=prefix['config'])
    raw = yaml.safe_load(full['config'].read_text())
    engines = load_config()['engine_configs']
    three = {
        **engines['three_level'],
        'calibration': {
            **engines['three_level']['calibration'],
            'population': 'prototype',
        },
    }
    (root / 'recal').mkdir()
    (root / 'recal' / 'three_level.yaml').write_text(yaml.safe_dump(three))
    (root / 'recal' / 'fixed_gates.yaml').write_text(
        yaml.safe_dump(engines['fixed_gates'])
    )
    raw['engines'] = {
        'fixed_gates': 'fixed_gates.yaml',
        'three_level': 'three_level.yaml',
    }
    raw['input']['residuals'] = str(full['table'])
    raw['input']['mask_file'] = str(full['tsf_mask'])
    config = root / 'recal' / 'ale.yaml'
    config.write_text(yaml.safe_dump(raw))
    run(
        'three_level',
        root / 'r2',
        config_path=config,
        state_dir=root / 'r1' / 'three_level' / 'state',
    )
    arrays, meta = read_alerts(root / 'r2' / 'three_level')
    previous_last = load_state(root / 'r1' / 'three_level' / 'state').last_date
    assert {'date': _next_date(arrays, previous_last), 'reason': 'recalibration'} in (
        meta['engine_attrs']['resets']
    )


def test_tampered_state_is_rejected(demo, capsys):
    root, prefix, full = demo
    run('three_level', root / 't1', config_path=prefix['config'])
    state_dir = root / 't1' / 'three_level' / 'state'
    payload = json.loads((state_dir / 'state.json').read_text())
    payload['tsf_cusum'][0] = (payload['tsf_cusum'][0] or 0.0) + 1.0
    (state_dir / 'state.json').write_text(json.dumps(payload))
    with pytest.raises(StateMismatchError, match='issued results would change'):
        run('three_level', root / 't2', config_path=full['config'], state_dir=state_dir)
    code = main(
        [
            'run',
            '--engine',
            'three_level',
            '--config',
            str(full['config']),
            '--out',
            str(root / 't3'),
            '--state',
            str(state_dir),
        ]
    )
    assert code == 1
    assert 'State error' in capsys.readouterr().err


def _next_date(arrays, last):
    """The first acquisition after ``last`` in an engine output."""
    return str(next(day for day in arrays['dates'] if day > last))
