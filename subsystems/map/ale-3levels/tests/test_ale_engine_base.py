"""Common output schema, severity mapping and configuration hashing."""

from __future__ import annotations

import numpy as np
import pytest
from ale_3levels.config import config_hash, load_config
from ale_3levels.engines.base import AlertResult, severity_from_levels

LEVELS = ('NO_DATA', 'WHITE', 'GREEN', 'YELLOW', 'RED')


def _result(**overrides):
    level = np.array([[[0, 1], [3, 4]]], dtype=np.int8)
    values = {
        'engine': 'fixed_gates',
        'engine_version': '0.1.0',
        'config_hash': 'x',
        'dates': np.array(['2020-01-01']),
        'levels': LEVELS,
        'level': level,
        'severity': severity_from_levels('fixed_gates', LEVELS, level),
        'metric_values': {'disp_30d_mm': np.zeros(level.shape)},
        'trigger_reasons': ('none',),
        'trigger_reason': np.zeros(level.shape, dtype=np.int16),
    }
    values.update(overrides)
    return AlertResult(**values)


def test_severity_mapping_is_documented_ordinal():
    result = _result()
    assert result.severity.tolist() == [[[-1, 0], [2, 3]]]
    assert result.labels().tolist() == [[['NO_DATA', 'WHITE'], ['YELLOW', 'RED']]]
    three = ('NO_DATA', 'NONE', 'L1', 'L2', 'L3')
    codes = np.arange(5, dtype=np.int8).reshape(1, 1, 5)
    assert severity_from_levels('three_level', three, codes).ravel().tolist() == [
        -1,
        0,
        1,
        2,
        3,
    ]


def test_result_rejects_inconsistent_severity():
    with pytest.raises(ValueError, match='severity'):
        _result(severity=np.zeros((1, 2, 2), dtype=np.int8))


def test_result_rejects_foreign_levels_and_shapes():
    with pytest.raises(ValueError, match='not native'):
        _result(levels=('NO_DATA', 'L1', 'GREEN', 'YELLOW', 'RED'))
    with pytest.raises(ValueError, match='disp_30d_mm'):
        _result(metric_values={'disp_30d_mm': np.zeros((2, 2))})


def test_default_config_loads_all_engine_files():
    config = load_config()
    assert set(config['engine_configs']) == {'fixed_gates', 'three_level'}
    gates = config['engine_configs']['fixed_gates']['gates']
    assert (gates['white_below'], gates['green_below'], gates['yellow_max']) == (
        1.0,
        4.0,
        10.0,
    )
    assert config['engine_configs']['three_level']['zscore_formula'] == 'frozen_sigma'


def test_config_hash_is_order_independent_and_sensitive():
    assert config_hash({'a': 1, 'b': [1, 2]}) == config_hash({'b': [1, 2], 'a': 1})
    assert config_hash({'a': 1}) != config_hash({'a': 1.5})
