import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from subsystems.dpr.preprocessing_pipelines.insar_diagnostics import InSARDiagnostics
from subsystems.dpr.preprocessing_pipelines.sentinel1 import Sentinel1Pipeline


def test_network_keeps_pair_identity_and_candidate_failures(tmp_path):
    diag = InSARDiagnostics(tmp_path, {})
    pairs = pd.DataFrame({'ref': ['2020-01-01', '2020-01-13'],
                          'rep': ['2020-01-13', '2020-01-25'],
                          'ref_baseline': [0., 10.], 'rep_baseline': [10., -5.]})
    diag.network(pairs, [{'days': 12, 'n_comp': 1, 'n_pairs': 2}])
    assert diag.summary['network'] == {'dates': 3, 'pairs': 2, 'single_edge_dates': 2}
    assert (diag.path / 'sbas_network.png').read_bytes().startswith(b'\x89PNG')
    assert json.loads((diag.path / 'sbas_pairs.json').read_text())[0]['rep'] == '2020-01-13'
    assert len(pd.read_csv(diag.path / 'acquisition_degree.csv')) == 3


def test_missing_phase_is_reported_and_json_is_strict(tmp_path):
    diag = InSARDiagnostics(tmp_path, {})
    phase = xr.DataArray([[[1., np.nan], [2., 3.]], [[np.nan]*2]*2],
                         dims=('pair', 'y', 'x'),
                         coords={'pair': ['a b', 'b c'], 'y': [0, 1], 'x': [0, 1]})
    corr = xr.where(np.isfinite(phase), .8, np.nan)
    diag.pairs(phase, corr, 'unwrapped_phase')
    rows = json.loads((diag.path / 'unwrapped_phase.json').read_text(),
                      parse_constant=lambda value: pytest.fail(f'Invalid JSON constant {value}'))
    assert [r['valid_phase_pixels'] for r in rows] == [3, 0]
    assert rows[1]['mean_coherence'] is None
    assert (diag.path / 'unwrapped_phase_sample_0001.png').is_file()


def test_failed_run_preserves_stage_and_earlier_artifacts(tmp_path):
    p = Sentinel1Pipeline()
    p._configure()
    p._config = {'result_dir': tmp_path}
    def broken_stage():
        p._diagnostics.table('earlier_result', pd.DataFrame({'count': [12]}))
        raise ValueError('No phase for 2020-01-13')
    p._run_processing = lambda: p._run_stage(broken_stage)
    p.close = Mock()
    with pytest.raises(ValueError, match='2020-01-13'):
        p._run()
    summary = json.loads((p._diagnostics.path / 'summary.json').read_text())
    assert summary['status'] == 'failed'
    assert summary['stages'][0]['status'] == 'failed'
    assert summary['stages'][0]['name'] == 'broken_stage'
    assert summary['stages'][0]['elapsed_seconds'] >= 0
    assert (p._diagnostics.path / 'earlier_result.csv').is_file()
    p.close.assert_called_once()


def test_reruns_have_distinct_reports_and_can_disable(tmp_path):
    first = InSARDiagnostics(tmp_path, {})
    second = InSARDiagnostics(tmp_path, {})
    assert first.path != second.path
    p = Sentinel1Pipeline()
    p._configure()
    p._config = {'result_dir': tmp_path, 'diagnostics': False}
    p._run_processing = lambda: 42
    assert p._run() == 42
    assert p._diagnostics is None


def test_displacement_statistics_have_units_in_column_names(tmp_path):
    diag = InSARDiagnostics(tmp_path, {})
    data = xr.DataArray([[[0., 0.]], [[-10., np.nan]]],
                        dims=('date', 'lat', 'lon'),
                        coords={'date': pd.to_datetime(['2020-01-01', '2020-01-13'])})
    diag.displacement(data)
    rows = pd.read_csv(diag.path / 'displacement_by_date.csv')
    assert rows.valid_fraction.tolist() == [1., .5]
    assert rows.mean_mm.tolist() == [0., -10.]


def test_reference_support_and_offset_are_saved(tmp_path):
    diag = InSARDiagnostics(tmp_path, {})
    report = pd.DataFrame({'reference_valid_phase_pixels': [10, 0],
                           'reference_mean_phase_rad': [1.2, np.nan]},
                          index=pd.Index(['a b', 'b c'], name='pair'))
    diag.reference(report)
    assert (diag.path / 'reference_phase.png').is_file()
    rows = json.loads((diag.path / 'reference_phase.json').read_text())
    assert rows[1]['reference_mean_phase_rad'] is None


def test_completed_run_is_recorded(tmp_path):
    p = Sentinel1Pipeline()
    p._configure()
    p._config = {'result_dir': tmp_path / 'results', 'diagnostics_root': str(tmp_path / 'site')}
    def successful_stage():
        return 42
    p._run_processing = lambda: p._run_stage(successful_stage)
    assert p._run() == 42
    assert p._diagnostics.path.parent == tmp_path / 'site' / 'diagnostics'
    assert not (tmp_path / 'results' / 'diagnostics').exists()
    summary = json.loads((p._diagnostics.path / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['stages'][0]['status'] == 'complete'
