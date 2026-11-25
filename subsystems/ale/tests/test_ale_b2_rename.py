"""B2.4: the L3 output is regime_change_score; P_regime is a legacy alias."""

from __future__ import annotations

import numpy as np
import pytest
from ale_3levels.outputs import get_metric, read_alerts
from ale_3levels.run import run
from ale_fixtures import write_long_table, write_run_config


def test_regime_change_score_and_its_aliases(tmp_path):
    table = write_long_table(tmp_path / 'table')
    run('both', tmp_path / 'out', config_path=write_run_config(tmp_path, table))
    arrays, metadata = read_alerts(tmp_path / 'out' / 'three_level')
    assert 'metric_regime_change_score' in arrays
    assert 'metric_P_regime' not in arrays
    assert metadata['metric_aliases'] == {
        'P_regime': 'regime_change_score',
        'regime_risk': 'regime_change_score',
    }
    score = get_metric(arrays, metadata, 'regime_change_score')
    np.testing.assert_array_equal(get_metric(arrays, metadata, 'P_regime'), score)
    a_arrays, a_metadata = read_alerts(tmp_path / 'out' / 'fixed_gates')
    assert a_metadata['metric_aliases'] == {}
    with pytest.raises(KeyError, match='P_regime'):
        get_metric(a_arrays, a_metadata, 'P_regime')
