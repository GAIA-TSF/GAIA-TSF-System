"""The demo-data script writes a small site that runs end to end."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'make_demo_data.py'


def _script():
    spec = importlib.util.spec_from_file_location('make_demo_data', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_site_runs_end_to_end(tmp_path):
    out = tmp_path / 'demo'
    assert (
        _script().main(['--out', str(out), '--size', '16', '--dates', '40', '--run'])
        == 0
    )
    for name in (
        'ale_demo.yaml',
        'README.txt',
        'results/forecasts/slope_trf/residuals.npz',
        'results/forecasts/slope_trf/residuals.json',
        'static/tsf_mask.tif',
        'static/roi_failure_mask.tif',
        'runs/demo/fixed_gates/alerts.npz',
        'runs/demo/three_level/alerts.npz',
        'report/report.md',
    ):
        assert (out / name).is_file(), name


def test_demo_refuses_existing_and_repository_paths(tmp_path):
    script = _script()
    out = tmp_path / 'demo'
    script.main(['--out', str(out), '--size', '16', '--dates', '40'])
    with pytest.raises(SystemExit):
        script.main(['--out', str(out)])
    assert (
        script.main(['--out', str(out), '--force', '--size', '16', '--dates', '40'])
        == 0
    )
    with pytest.raises(SystemExit):
        script.main(['--out', str(SCRIPT.parents[1] / 'demo')])
    other = tmp_path / 'unrelated'
    other.mkdir()
    (other / 'keep.txt').write_text('x')
    with pytest.raises(SystemExit):
        script.main(['--out', str(other), '--force'])
    assert (other / 'keep.txt').is_file()
