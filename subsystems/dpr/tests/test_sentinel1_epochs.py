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


def test_dates_with_missing_bursts_are_excluded():
    p = Sentinel1Pipeline(); p._configure()
    scenes = pd.DataFrame({
        'datetime': ['2026-03-04 18:26:05', '2026-03-04 18:26:08', '2026-03-04 18:26:05',
                     '2026-04-15 18:27:07',
                     '2026-04-21 18:26:07', '2026-04-21 18:26:10'],
        'subswath': [2, 2, 2, 2, 2, 2],
        'datapath': list('abcdef')},
        index=pd.Index(['2026-03-04', '2026-03-04', '2026-03-04', '2026-04-15', '2026-04-21', '2026-04-21'],
                       name='date'))
    kept = p._filter_incomplete_dates(scenes)
    # the duplicate 'c' shares a start time with 'a' and does not count as an extra burst
    assert kept.datapath.tolist() == list('abcef')


def test_too_few_complete_dates_fail():
    p = Sentinel1Pipeline(); p._configure()
    scenes = pd.DataFrame({'datetime': ['t1', 't2', 't3'], 'subswath': [2, 2, 2]},
                          index=['2026-03-04', '2026-03-04', '2026-04-15'])
    with pytest.raises(ValueError, match='complete burst coverage'):
        p._filter_incomplete_dates(scenes)


def test_other_workdir_scenes_are_not_inputs(monkeypatch, tmp_path):
    from subsystems.dpr.preprocessing_pipelines import sentinel1
    p = Sentinel1Pipeline(); p._configure()
    p._config = {'source_safe_only': True, 'excluded_dates': [],
                 'processing_start': '2018-03-21', 'processing_end': '2018-12-31'}
    scenes = pd.DataFrame({'datapath': ['/data/a.SAFE/a.tiff', '/data/b.SAFE/b.tiff',
                                       '/data/workdir/a.tiff', '/data/c.SAFE/c.tiff']},
                          index=['2018-03-21', '2018-04-02', '2018-03-21', '2018-02-25'])
    monkeypatch.setattr(p, '_scan_original_slcs', lambda path: scenes)
    stack = Mock(); stack.set_scenes.return_value = stack
    monkeypatch.setattr(sentinel1, 'Stack', lambda *a, **kw: stack)
    p._stack_scenes(tmp_path / 'inputs', tmp_path / 'work')
    selected = stack.set_scenes.call_args.args[0]
    assert selected.datapath.tolist() == ['/data/a.SAFE/a.tiff', '/data/b.SAFE/b.tiff']


@pytest.mark.parametrize('value', [None, '', 'None', 'NaT', 'not-a-date'])
def test_empty_or_invalid_epoch_dates_rejected(value):
    c = config()
    c['sentinel1']['epochs']['post_failure']['start'] = value
    with pytest.raises(ValueError, match='sentinel1.epochs.post_failure.start'):
        processing_runs(c)


def test_missing_monitoring_end_rejected_even_in_full_mode():
    c = config()
    del c['project']['monitoring_period']['end']
    with pytest.raises(ValueError, match='project.monitoring_period.end'):
        processing_runs(c, 'full')


def test_pre_failure_only_configuration():
    c = config()
    del c['sentinel1']['epochs']['post_failure']
    assert [r['name'] for r in processing_runs(c)] == ['pre_failure']


def test_original_scan_hides_generated_basename_collisions(tmp_path, monkeypatch):
    from subsystems.dpr.preprocessing_pipelines import sentinel1
    source = tmp_path / 'original.SAFE' / 'measurement'
    source.mkdir(parents=True)
    (source / 'burst.tiff').touch()
    generated = tmp_path / 'old_workdir'
    generated.mkdir()
    (generated / 'burst.tiff').touch()
    orbit = tmp_path / 'orbit.EOF'
    orbit.touch()
    def scan(view):
        import glob
        files = glob.glob(str(view / '**' / '*.tiff'), recursive=True)
        assert len(files) == 1
        assert Path(files[0]).resolve() == source / 'burst.tiff'
        orbits = glob.glob(str(view / '**' / '*.EOF'), recursive=True)
        return pd.DataFrame({'datapath': files, 'metapath': files, 'orbitpath': orbits})
    monkeypatch.setattr(sentinel1.S1, 'scan_slc', scan)
    scenes = Sentinel1Pipeline._scan_original_slcs(tmp_path)
    assert scenes.datapath.iloc[0] == str(source / 'burst.tiff')
    assert Path(scenes.orbitpath.iloc[0]).exists()
