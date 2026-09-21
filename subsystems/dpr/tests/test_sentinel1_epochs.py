from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from subsystems.dpr.preprocessing_pipelines.sentinel1 import Sentinel1Pipeline
from subsystems.dpr.run_s1_eou_dpr import processing_runs


def config():
    return {'project': {'data_dir': '/site', 'monitoring_period': {'start': '2015-01-01', 'end': '2018-12-31'}},
            'sentinel1': {'epochs': {'pre_failure': {'start': '2015-01-01', 'end': '2018-02-25'},
                                    'post_failure': {'start': '2018-03-21', 'end': '2018-12-31'}}}}


def test_independent_paths_and_default_split():
    runs = processing_runs(config())
    assert [r['name'] for r in runs] == ['pre_failure', 'post_failure']
    assert runs[0]['workdir'] != runs[1]['workdir']
    assert runs[0]['result_dir'] == Path('/site/sentinel1/results_epochs/pre_failure')
    assert runs[1]['start'] == '2018-03-21'
    assert all(Path('/site/sentinel1') not in r['workdir'].parents for r in runs)
    custom = processing_runs(config(), 'all', Path('/custom'))
    assert custom[1]['result_dir'] == Path('/custom/post_failure')


def test_overlap_rejected_before_running():
    c = config()
    c['sentinel1']['epochs']['post_failure']['start'] = '2018-02-25'
    with pytest.raises(ValueError, match='overlap'):
        processing_runs(c)


def test_non_epoch_project_and_explicit_full():
    c = config()
    c['sentinel1'] = {}
    assert processing_runs(c)[0]['name'] == 'full'
    assert processing_runs(config(), 'full')[0]['end'] == '2018-12-31'
    with pytest.raises(ValueError):
        processing_runs(c, 'pre_failure')


def test_local_filter_inclusive_dates_keeps_all_bursts():
    p = Sentinel1Pipeline(); p._configure()
    scenes = pd.DataFrame({'datapath': list('abcdef')},
                          index=['2018-02-13', '2018-02-25', '2018-02-25', '2018-03-09', '2018-03-21', '2018-04-02'])
    p._config = {'processing_start': '2015-01-01', 'processing_end': '2018-02-25'}
    pre = p._filter_processing_dates(scenes)
    assert pre.datapath.tolist() == list('abc')
    p._config = {'processing_start': '2018-03-21', 'processing_end': '2018-12-31'}
    post = p._filter_processing_dates(scenes)
    assert post.datapath.tolist() == list('ef')
    assert set(pre.index).isdisjoint(post.index)


def test_single_date_fails_even_with_multiple_bursts():
    p = Sentinel1Pipeline(); p._configure()
    p._config = {'processing_start': '2018-02-25', 'processing_end': '2018-02-25'}
    with pytest.raises(ValueError, match='two acquisition'):
        p._filter_processing_dates(pd.DataFrame(index=['2018-02-25', '2018-02-25']))


def test_other_workdir_scenes_are_not_inputs(monkeypatch, tmp_path):
    from subsystems.dpr.preprocessing_pipelines import sentinel1
    p = Sentinel1Pipeline(); p._configure()
    p._config = {'source_safe_only': True, 'excluded_dates': [],
                 'processing_start': '2018-03-21', 'processing_end': '2018-12-31'}
    scenes = pd.DataFrame({'datapath': ['/data/a.SAFE/a.tiff', '/data/b.SAFE/b.tiff',
                                       '/data/workdir/a.tiff', '/data/c.SAFE/c.tiff']},
                          index=['2018-03-21', '2018-04-02', '2018-03-21', '2018-02-25'])
    monkeypatch.setattr(sentinel1.S1, 'scan_slc', lambda path: scenes)
    stack = Mock(); stack.set_scenes.return_value = stack
    monkeypatch.setattr(sentinel1, 'Stack', lambda *a, **kw: stack)
    p._stack_scenes(tmp_path / 'inputs', tmp_path / 'work')
    selected = stack.set_scenes.call_args.args[0]
    assert selected.datapath.tolist() == ['/data/a.SAFE/a.tiff', '/data/b.SAFE/b.tiff']
