"""Engine A (fixed_gates): gates, boundaries, cadence, doubling, residual input."""

from __future__ import annotations

import builtins
import os
import socket
import subprocess

import numpy as np
import pytest
from ale_3levels.config import load_config
from ale_3levels.engines.base import Engine
from ale_3levels.engines.fixed_gates import (
    LEVELS,
    TRIGGER_REASONS,
    FixedGatesEngine,
)
from ale_fixtures import iso, make_contract


def _config(**overrides):
    config = dict(load_config()['engine_configs']['fixed_gates'])
    for key, value in overrides.items():
        if isinstance(value, dict):
            config[key] = {**config.get(key, {}), **value}
        else:
            config[key] = value
    return config


def _score(days, values, **overrides):
    """Score a one-pixel series given as (day offsets, cumulative mm)."""
    series = np.asarray(values, dtype=float).reshape(-1, 1, 1)
    contract = make_contract([iso(d) for d in days], series)
    return FixedGatesEngine(_config(**overrides)).score(contract)


def _labels(result):
    return result.labels()[:, 0, 0].tolist()


def _reason(result, index=-1):
    return TRIGGER_REASONS[int(result.trigger_reason[index, 0, 0])]


BOUNDARIES = [
    (0.99, 'WHITE'),
    (1.00, 'GREEN'),
    (3.99, 'GREEN'),
    (4.00, 'YELLOW'),
    (10.00, 'YELLOW'),
    (10.01, 'RED'),
]


@pytest.mark.parametrize(('movement', 'expected'), BOUNDARIES)
@pytest.mark.parametrize(
    ('direction', 'sign'),
    [('abs', -1.0), ('abs', 1.0), ('negative', -1.0), ('positive', 1.0)],
)
def test_gate_boundaries(movement, expected, direction, sign):
    result = _score([0, 30], [0.0, sign * movement], direction=direction)
    assert _labels(result) == [expected]
    assert result.metric_values['disp_30d_mm'][0, 0, 0] == pytest.approx(
        sign * movement
    )


@pytest.mark.parametrize(('direction', 'sign'), [('negative', 1.0), ('positive', -1.0)])
def test_opposite_direction_is_white(direction, sign):
    result = _score([0, 30], [0.0, sign * 25.0], direction=direction)
    assert _labels(result) == ['WHITE']


def test_negative_means_motion_away_from_the_satellite():
    """LOS sign convention: settlement is negative LOS."""
    result = _score([0, 30], [0.0, -12.0], direction='negative')
    assert _labels(result) == ['RED']
    assert _reason(result).startswith('red_threshold|')


@pytest.mark.parametrize(
    ('previous', 'current', 'level', 'status'),
    [
        (2.5, 5.0, 'RED', 'doubling_fired'),  # 2x at/above 4 mm
        (3.0, 5.0, 'YELLOW', 'doubling_not_met'),  # 1.67x
        (1.5, 3.0, 'GREEN', 'doubling_below_min_rate'),  # 2x below 4 mm
        (0.5, 5.0, 'YELLOW', 'doubling_prior_below_floor'),  # near-zero prior
        (4.0, 8.0, 'RED', 'doubling_fired'),
    ],
)
def test_velocity_doubling(previous, current, level, status):
    result = _score([0, 30, 60], [0.0, -previous, -previous - current])
    assert _labels(result)[-1] == level
    assert _reason(result).endswith(status)
    assert result.metric_values['velocity_ratio'][-1, 0, 0] == pytest.approx(
        current / previous
    )


def test_doubling_can_be_disabled():
    result = _score(
        [0, 30, 60], [0.0, -2.5, -7.5], velocity_doubling={'enabled': False}
    )
    assert _labels(result)[-1] == 'YELLOW'
    assert _reason(result).endswith('doubling_disabled')


def test_doubling_skipped_without_prior_window_keeps_the_gate():
    result = _score([0, 30], [0.0, -5.0])
    assert _labels(result) == ['YELLOW']
    assert _reason(result) == 'yellow|doubling_skipped_no_prior_window'


def test_irregular_cadence_interpolates_the_window_start():
    # 6/12/24/12-day gaps. At t=54, t-30=24 lies between days 18 and 42.
    days = [0, 6, 18, 42, 54]
    values = [0.0, -0.5, -1.0, -4.0, -6.0]
    result = _score(days, values)
    expected_start = -1.0 + (24 - 18) / (42 - 18) * (-4.0 - -1.0)
    assert result.metric_values['disp_30d_mm'][-1, 0, 0] == pytest.approx(
        -6.0 - expected_start
    )
    # At t=42, t-30=12 lies between days 6 and 18.
    start_42 = -0.5 + (12 - 6) / 12 * (-1.0 - -0.5)
    assert result.metric_values['disp_30d_mm'][2, 0, 0] == pytest.approx(
        -4.0 - start_42
    )


def test_no_data_when_no_acquisition_within_tolerance():
    # t=54: the acquisition before t-30=24 is day 18, 6 days earlier.
    days, values = [0, 6, 18, 42, 54], [0.0, -0.5, -1.0, -4.0, -6.0]
    strict = _score(days, values, interpolation_tolerance_days=5)
    assert _labels(strict)[-1] == 'NO_DATA'
    assert _reason(strict) == 'no_data:no_acquisition_within_tolerance'
    assert np.isnan(strict.severity[-1, 0, 0]) or strict.severity[-1, 0, 0] == -1
    # A 60-day gap: t-30 is 30 days after the only earlier acquisition.
    gap = _score([0, 60], [0.0, -20.0])
    assert _labels(gap) == ['NO_DATA']
    # Before the first acquisition there is no window at all.
    early = _score([0, 12, 24], [0.0, -1.0, -2.0])
    assert _labels(early) == ['NO_DATA', 'NO_DATA']


def test_no_data_for_invalid_and_missing_pixels():
    days = [0, 30, 60]
    series = np.zeros((3, 1, 3))
    series[:, 0, 0] = [0.0, -5.0, -6.0]
    series[:, 0, 1] = [0.0, -5.0, -6.0]
    series[1, 0, 2] = np.nan  # missing at a bracketing acquisition
    mask = np.array([[True, False, True]])
    contract = make_contract([iso(d) for d in days], series, mask=mask)
    result = FixedGatesEngine(_config()).score(contract)
    labels = result.labels()
    assert labels[:, 0, 0].tolist() == ['YELLOW', 'GREEN']
    assert labels[:, 0, 1].tolist() == ['NO_DATA', 'NO_DATA']
    assert TRIGGER_REASONS[result.trigger_reason[0, 0, 1]] == 'no_data:outside_mask'
    assert labels[:, 0, 2].tolist() == ['NO_DATA', 'NO_DATA']
    assert TRIGGER_REASONS[result.trigger_reason[0, 0, 2]] == 'no_data:missing_value'
    assert np.all(np.isnan(result.metric_values['disp_30d_mm'][:, 0, 1]))


def test_residual_input_equals_observed_when_residual_is_the_increment():
    """Persistence forecast: residual = observed increment, so both inputs agree."""
    days = [0, 6, 18, 42, 54, 66, 72]
    values = np.array([0.0, -0.5, -1.0, -4.0, -6.0, -9.0, -15.0]).reshape(-1, 1, 1)
    contract = make_contract([iso(d) for d in days], values)
    observed = FixedGatesEngine(_config(gate_input='observed')).score(contract)
    residual = FixedGatesEngine(_config(gate_input='residual')).score(contract)
    np.testing.assert_array_equal(observed.level, residual.level)
    np.testing.assert_allclose(
        observed.metric_values['disp_30d_mm'],
        residual.metric_values['disp_30d_mm'],
        rtol=1e-12,
    )


def test_residual_input_ignores_movement_the_model_expected():
    days = [0, 12, 24, 36, 48]
    values = np.array([0.0, -6.0, -12.0, -18.0, -24.0]).reshape(-1, 1, 1)
    perfect = make_contract(
        [iso(d) for d in days], values, residual_mm=np.zeros((4, 1, 1))
    )
    observed = FixedGatesEngine(_config()).score(perfect)
    residual = FixedGatesEngine(_config(gate_input='residual')).score(perfect)
    assert _labels(observed)[-1] == 'RED'
    assert _labels(residual)[-1] == 'WHITE'


def test_result_schema_and_actions():
    result = _score([0, 30, 60], [0.0, -2.5, -7.5])
    engine = FixedGatesEngine(_config())
    assert isinstance(engine, Engine)
    assert result.engine == 'fixed_gates'
    assert result.levels == LEVELS
    assert result.severity[:, 0, 0].tolist() == [1, 3]  # GREEN (2.5 mm), RED (doubling)
    assert result.config_hash == engine.config_hash
    assert set(result.metric_values) >= {
        'disp_30d_mm',
        'rate_mm_per_month',
        'velocity_ratio',
    }
    assert result.recommended_actions['RED'][0].startswith('Halt heavy operations')
    assert result.attrs['level_scope'] == 'pixel'


def test_window_and_rate_scaling():
    result = _score([0, 15], [0.0, -3.0], window_days=15)
    assert result.metric_values['disp_30d_mm'][0, 0, 0] == pytest.approx(-3.0)
    assert result.metric_values['rate_mm_per_month'][0, 0, 0] == pytest.approx(-6.0)
    assert _labels(result) == ['YELLOW']


@pytest.mark.parametrize(
    'overrides',
    [
        {'gate_input': 'other'},
        {'direction': 'up'},
        {'gates': {'white_below': 5.0}},
        {'velocity_doubling': {'ratio': 1.0}},
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        FixedGatesEngine(_config(**overrides))


def test_engine_has_no_side_effects(tmp_path, monkeypatch):
    """No files, network, processes or shell commands while scoring."""
    contract = make_contract(
        [iso(d) for d in (0, 30, 60)],
        np.array([0.0, -2.5, -20.0]).reshape(-1, 1, 1),
    )
    engine = FixedGatesEngine(_config())

    def forbidden(*args, **kwargs):
        raise AssertionError(f'side effect attempted: {args!r}')

    real_open = builtins.open

    def guarded_open(file, mode='r', *args, **kwargs):
        if any(flag in mode for flag in 'wax+'):
            forbidden(file, mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(os, 'system', forbidden)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    result = engine.score(contract)
    assert result.labels()[-1, 0, 0] == 'RED'
    assert list(tmp_path.iterdir()) == []
