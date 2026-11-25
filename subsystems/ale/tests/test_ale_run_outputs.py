"""`run` writes both engines in the common schema, once and reproducibly."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from ale_3levels.__main__ import main
from ale_3levels.outputs import read_alerts
from ale_3levels.run import run
from ale_fixtures import write_long_table, write_run_config

ENGINE_FILES = {
    'alerts.json',
    'alerts.npz',
    'history.csv',
    'sectors.csv',
    'sectors.json',
}


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp('run')
    table = write_long_table(root / 'site' / 'results' / 'forecasts' / 'slope_trf')
    config = write_run_config(root, table)
    out = root / 'out'
    run('both', out, config_path=config)
    return root, table, config, out


def _files(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob('*'))
        if path.is_file()
    }


def test_both_engines_write_the_common_schema(setup):
    _, _, _, out = setup
    assert (out / 'run.json').is_file()
    shapes = {}
    for engine in ('fixed_gates', 'three_level'):
        names = {path.name for path in (out / engine).iterdir() if path.is_file()}
        assert names == ENGINE_FILES
        arrays, metadata = read_alerts(out / engine)
        assert metadata['engine'] == engine
        assert metadata['engine_version'] and len(metadata['config_hash']) == 64
        assert metadata['input']['model_version'] == 'slope_trf@long'
        assert metadata['severity_mapping']['NO_DATA'] == -1
        assert 'comparison only' in metadata['severity_note']
        for key in ('level', 'severity', 'trigger_reason', 'level_scope', 'region_id'):
            assert arrays[key].shape == arrays['level'].shape
        assert set(np.unique(arrays['severity'])) <= {-1, 0, 1, 2, 3}
        shapes[engine] = arrays['level'].shape
    assert shapes['fixed_gates'] == shapes['three_level']


def test_engine_specific_metrics_and_actions(setup):
    _, _, _, out = setup
    a, meta_a = read_alerts(out / 'fixed_gates')
    b, meta_b = read_alerts(out / 'three_level')
    assert {
        'metric_disp_30d_mm',
        'metric_rate_mm_per_month',
        'metric_velocity_ratio',
    } <= set(a)
    assert {'metric_z', 'metric_cusum_pos', 'metric_cusum_neg'} <= set(b)
    assert 'metric_regime_change_score' in b
    assert meta_a['recommended_actions']['RED']
    assert meta_b['level_scopes'] == ['pixel', 'region']


def test_history_and_sector_tables(setup):
    _, _, _, out = setup
    for engine in ('fixed_gates', 'three_level'):
        arrays, metadata = read_alerts(out / engine)
        with (out / engine / 'history.csv').open() as stream:
            history = list(csv.DictReader(stream))
        assert [row['date'] for row in history] == arrays['dates'].tolist()
        for index, row in enumerate(history):
            counts = sum(int(row[f'pixels_{label}']) for label in metadata['levels'])
            assert counts == int(arrays['mask'].sum())
        with (out / engine / 'sectors.csv').open() as stream:
            sectors = list(csv.DictReader(stream))
        assert {row['sector_id'] for row in sectors} == {'tsf'}
        assert len(sectors) == len(history)
        assert {row['level'] for row in sectors} <= set(metadata['levels'])


def test_outputs_are_byte_identical_for_the_same_input(setup, tmp_path):
    _, _, config, out = setup
    again = tmp_path / 'again'
    run('both', again, config_path=config)
    assert _files(again) == _files(out)


def test_outputs_are_written_once(setup):
    _, _, config, out = setup
    with pytest.raises(FileExistsError):
        run('fixed_gates', out, config_path=config)


def test_run_does_not_touch_map_outputs(setup, tmp_path):
    _, table, config, _ = setup
    before = _files(table.parent)
    run('fixed_gates', tmp_path / 'other', config_path=config)
    assert _files(table.parent) == before


def test_cli_run_single_engine(setup, tmp_path, capsys):
    _, _, config, _ = setup
    code = main(
        [
            'run',
            '--engine',
            'fixed_gates',
            '--config',
            str(config),
            '--out',
            str(tmp_path / 'cli'),
        ]
    )
    assert code == 0
    assert (tmp_path / 'cli' / 'fixed_gates' / 'alerts.npz').is_file()
    assert not (tmp_path / 'cli' / 'three_level').exists()
    record = json.loads((tmp_path / 'cli' / 'run.json').read_text())
    assert record['engines'] == ['fixed_gates']
