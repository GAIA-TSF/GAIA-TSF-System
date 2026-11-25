"""Comparison report, control area, and the ground-truth isolation rule."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from ale_3levels.compare import compare, control_area
from ale_3levels.config import DEFAULT_CONFIG
from ale_3levels.run import run
from ale_fixtures import write_long_table, write_run_config

PACKAGE = Path(__file__).resolve().parents[1] / 'ale_3levels'
PATCH = (slice(2, 7), slice(2, 7))


@pytest.fixture(scope='module')
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp('compare')
    table = write_long_table(
        root / 'site' / 'results' / 'forecasts' / 'slope_trf',
        height=20,
        width=20,
        patch=PATCH,
    )
    config = write_run_config(root, table)
    run('both', root / 'out', config_path=config)
    truth = np.zeros((20, 20), dtype='uint8')
    truth[PATCH] = 1
    truth_path = root / 'roi_failure_mask.tif'
    with rasterio.open(
        truth_path,
        'w',
        driver='GTiff',
        height=20,
        width=20,
        count=1,
        dtype='uint8',
        crs='EPSG:32735',
        transform=Affine(10, 0, 500000, 0, -10, 7000000),
    ) as raster:
        raster.write(truth, 1)
    paths = compare(
        root / 'out' / 'fixed_gates',
        root / 'out' / 'three_level',
        root / 'report',
        truth=truth_path,
    )
    return root, paths


def test_report_and_tables_are_written(report):
    _, paths = report
    for name in (
        'report',
        'summary',
        'timeline',
        'confusion_pixel',
        'confusion_sector',
        'alert_counts',
        'first_alerts',
        'flips',
        'ground_truth',
    ):
        assert paths[name].is_file(), name
    text = paths['report'].read_text()
    for heading in ('Run identity', 'Ground truth', 'confusion matrix', 'Noise near'):
        assert heading in text
    summary = json.loads(paths['summary'].read_text())
    assert summary['input']['model_version'] == 'slope_trf@long'
    assert summary['engine_b_calibration']['calibration_dates'] > 0
    assert set(summary['engines']) == {'fixed_gates', 'three_level'}
    for engine in summary['engines'].values():
        assert len(engine['config_hash']) == 64


def test_confusion_matrix_counts_every_cell(report):
    _, paths = report
    summary = json.loads(paths['summary'].read_text())
    pixel = np.array(summary['pixel_confusion'])
    assert pixel.sum() == summary['monitoring_dates'] * 400
    sector = np.array(summary['sector_confusion'])
    assert sector.sum() == summary['monitoring_dates']


def test_ground_truth_rows_cover_both_areas(report):
    _, paths = report
    with paths['ground_truth'].open() as stream:
        rows = list(csv.DictReader(stream))
    assert {(r['engine'], r['area']) for r in rows} == {
        (e, a) for e in ('fixed_gates', 'three_level') for a in ('failure', 'control')
    }
    truth = json.loads(paths['summary'].read_text())['ground_truth']
    assert truth['failure_pixels'] == truth['control_pixels'] == 25
    # The accelerating patch is flagged by engine A at some level.
    assert truth['engines']['fixed_gates']['first_failure_alert'] is not None


def test_control_area_is_equal_sized_disjoint_and_buffered():
    mask = np.ones((30, 30), dtype=bool)
    failure = np.zeros_like(mask)
    failure[2:7, 2:7] = True
    control = control_area(mask, failure, buffer=5)
    assert control.sum() == failure.sum()
    assert not np.any(control & failure)
    rows, cols = np.nonzero(control)
    assert min(rows.min() - 6, cols.min() - 6) > 0  # beyond the 5-pixel buffer
    np.testing.assert_array_equal(control_area(mask, failure, buffer=5), control)


def test_compare_rejects_mismatched_engines(report, tmp_path):
    root, _ = report
    with pytest.raises(ValueError, match='--a must be'):
        compare(root / 'out' / 'three_level', root / 'out' / 'fixed_gates', tmp_path)


def test_ground_truth_is_read_only_by_the_report():
    """No engine, calibration or config may use the ground-truth mask."""
    pattern = re.compile(r'roi_failure|truth', re.IGNORECASE)
    offenders = [
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob('*.py')
        if path.name not in {'compare.py', '__main__.py'}
        and pattern.search(path.read_text())
    ]
    assert offenders == []
    for config in DEFAULT_CONFIG.parent.glob('*.yaml'):
        assert not pattern.search(config.read_text()), config.name
