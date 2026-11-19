from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from subsystems.dpr.preprocessing_pipelines.sentinel1 import Sentinel1Pipeline
from lib.exceptions import GaiaDataError


def test_local_filter_inclusive_dates_keeps_all_bursts():
    p = Sentinel1Pipeline()
    p._configure()
    scenes = pd.DataFrame(
        {'datapath': list('abcdef')},
        index=[
            '2018-02-13',
            '2018-02-25',
            '2018-02-25',
            '2018-03-09',
            '2018-03-21',
            '2018-04-02',
        ],
    )
    p._config = {'processing_start': '2015-01-01', 'processing_end': '2018-02-25'}
    pre = p._filter_processing_dates(scenes)
    assert pre.datapath.tolist() == list('abc')
    p._config = {'processing_start': '2018-03-21', 'processing_end': '2018-12-31'}
    post = p._filter_processing_dates(scenes)
    assert post.datapath.tolist() == list('ef')
    assert set(pre.index).isdisjoint(post.index)


def test_single_date_fails_even_with_multiple_bursts():
    p = Sentinel1Pipeline()
    p._configure()
    p._config = {'processing_start': '2018-02-25', 'processing_end': '2018-02-25'}
    with pytest.raises(GaiaDataError, match='two acquisition'):
        p._filter_processing_dates(pd.DataFrame(index=['2018-02-25', '2018-02-25']))


def test_dates_with_missing_bursts_are_excluded():
    p = Sentinel1Pipeline()
    p._configure()
    scenes = pd.DataFrame(
        {
            'datetime': [
                '2026-03-04 18:26:05',
                '2026-03-04 18:26:08',
                '2026-03-04 18:26:05',
                '2026-04-15 18:27:07',
                '2026-04-21 18:26:07',
                '2026-04-21 18:26:10',
            ],
            'subswath': [2, 2, 2, 2, 2, 2],
            'datapath': list('abcdef'),
        },
        index=pd.Index(
            [
                '2026-03-04',
                '2026-03-04',
                '2026-03-04',
                '2026-04-15',
                '2026-04-21',
                '2026-04-21',
            ],
            name='date',
        ),
    )
    kept = p._filter_incomplete_dates(scenes)
    # the duplicate 'c' shares a start time with 'a' and does not count as an extra burst
    assert kept.datapath.tolist() == list('abcef')


def test_too_few_complete_dates_fail():
    p = Sentinel1Pipeline()
    p._configure()
    scenes = pd.DataFrame(
        {'datetime': ['t1', 't2', 't3'], 'subswath': [2, 2, 2]},
        index=['2026-03-04', '2026-03-04', '2026-04-15'],
    )
    with pytest.raises(GaiaDataError, match='complete burst coverage'):
        p._filter_incomplete_dates(scenes)


def test_stacking_scans_only_original_safe_scenes(monkeypatch, tmp_path):
    from subsystems.dpr.preprocessing_pipelines import sentinel1

    p = Sentinel1Pipeline()
    p._configure()
    p._config = {
        'excluded_dates': [],
        'processing_start': '2018-03-21',
        'processing_end': '2018-12-31',
    }
    scenes = pd.DataFrame(
        {
            'datapath': [
                '/data/a.SAFE/a.tiff',
                '/data/b.SAFE/b.tiff',
                '/data/c.SAFE/c.tiff',
            ]
        },
        index=['2018-03-21', '2018-04-02', '2018-02-25'],
    )
    monkeypatch.setattr(p, '_scan_original_slcs', lambda *args, **kwargs: scenes)

    def plain_scan(*args, **kwargs):
        raise AssertionError('datadir must not be scanned directly')

    monkeypatch.setattr(sentinel1.S1, 'scan_slc', plain_scan)
    stack = Mock()
    stack.set_scenes.return_value = stack
    monkeypatch.setattr(sentinel1, 'Stack', lambda *a, **kw: stack)
    p._stack_scenes(tmp_path / 'inputs', tmp_path / 'work')
    selected = stack.set_scenes.call_args.args[0]
    assert selected.datapath.tolist() == ['/data/a.SAFE/a.tiff', '/data/b.SAFE/b.tiff']


def test_original_scan_hides_generated_basename_collisions(tmp_path, monkeypatch):
    from subsystems.dpr.preprocessing_pipelines import sentinel1

    source = tmp_path / 'original.SAFE' / 'measurement'
    source.mkdir(parents=True)
    (source / 'burst.tiff').touch()
    # InSAR needs at least two scenes
    other = tmp_path / 'other.SAFE' / 'measurement'
    other.mkdir(parents=True)
    (other / 'other.tiff').touch()
    generated = tmp_path / 'old_workdir'
    generated.mkdir()
    (generated / 'burst.tiff').touch()
    orbit = tmp_path / 'orbit.EOF'
    orbit.touch()

    def scan(view):
        import glob

        files = sorted(glob.glob(str(view / '**' / '*.tiff'), recursive=True))
        assert {Path(f).resolve() for f in files} == {
            source / 'burst.tiff',
            other / 'other.tiff',
        }
        orbits = glob.glob(str(view / '**' / '*.EOF'), recursive=True)
        assert len(orbits) == 1
        return pd.DataFrame(
            {'datapath': files, 'metapath': files, 'orbitpath': orbits * len(files)}
        )

    monkeypatch.setattr(sentinel1.S1, 'scan_slc', scan)
    scenes = Sentinel1Pipeline._scan_original_slcs(tmp_path)
    assert set(scenes.datapath) == {
        str(source / 'burst.tiff'),
        str(other / 'other.tiff'),
    }
    assert all(Path(path).exists() for path in scenes.orbitpath)


def make_burst_safe(root, mission, stamp, subswath):
    """Create a minimal ASF burst SAFE with one measurement/annotation pair."""
    safe = (
        root
        / f'{mission}_IW_SLC__1SDV_{stamp}T063532_{stamp}T063535_063474_07F990_26C4.SAFE'
    )
    name = f'{mission.lower()}-iw{subswath}-slc-vv-{stamp}t063532-{stamp}t063535-063474-07f990-001'
    (safe / 'measurement').mkdir(parents=True)
    (safe / 'annotation').mkdir()
    (safe / 'measurement' / f'{name}.tiff').touch()
    (safe / 'annotation' / f'{name}.xml').touch()
    return safe


def test_original_scan_limited_to_processing_window(tmp_path, monkeypatch):
    """Epochs share datadir; another epoch's track must not reach scan_slc."""
    from subsystems.dpr.preprocessing_pipelines import sentinel1

    make_burst_safe(tmp_path, 'S1A', '20260110', 3)  # pre_failure, track 154
    make_burst_safe(tmp_path, 'S1A', '20260215', 3)
    inside = [
        make_burst_safe(tmp_path, 'S1A', '20260304', 1),  # post_failure, track 52
        make_burst_safe(tmp_path, 'S1C', '20260310', 1),
    ]

    def scan(view):
        import glob

        files = glob.glob(str(view / '**' / '*.tiff'), recursive=True)
        return pd.DataFrame({'datapath': files, 'metapath': files})

    monkeypatch.setattr(sentinel1.S1, 'scan_slc', scan)
    scenes = Sentinel1Pipeline._scan_original_slcs(tmp_path, '2026-03-01', '2026-10-29')
    assert sorted(Path(path).parents[1] for path in scenes.datapath) == inside


def test_original_scan_rejects_mixed_tracks_in_window(tmp_path, monkeypatch):
    from subsystems.dpr.preprocessing_pipelines import sentinel1

    make_burst_safe(tmp_path, 'S1A', '20260215', 3)
    make_burst_safe(tmp_path, 'S1A', '20260304', 1)

    def scan(view):
        raise AssertionError('mixed subswaths must be rejected before scan_slc')

    monkeypatch.setattr(sentinel1.S1, 'scan_slc', scan)
    with pytest.raises(
        GaiaDataError, match=r'IW1: 1 SAFEs \(20260304\.\.20260304\).*path_number'
    ):
        Sentinel1Pipeline._scan_original_slcs(tmp_path, '2026-01-01', '2026-10-29')


def test_original_scan_empty_processing_window(tmp_path):
    make_burst_safe(tmp_path, 'S1A', '20260215', 3)
    with pytest.raises(GaiaDataError, match='between 2026-03-01 and 2026-10-29'):
        Sentinel1Pipeline._scan_original_slcs(tmp_path, '2026-03-01', '2026-10-29')
