"""Showcase demo site, engine aliases A/B, and the run_demo.py script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ale_3levels.compare import control_area
from ale_3levels.config import load_config
from ale_3levels.contract import summary
from ale_3levels.demo import SHOWCASE_ZONES, showcase_layout, simulate_showcase
from ale_3levels.run import load_input, run

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_demo.py'


def _script():
    spec = importlib.util.spec_from_file_location('run_demo', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def demo_run(tmp_path_factory):
    """The script end to end on the quick site, both engines, no plots."""
    out = tmp_path_factory.mktemp('showcase') / 'site'
    assert _script().main(['--out', str(out), '--quick', '--no-plots']) == 0
    return out


def test_layout_zones_are_disjoint_and_inside_the_tsf():
    layout = showcase_layout()
    zones, mask = layout['zones'], layout['mask']
    assert np.all((zones > 0) == mask)
    assert not np.any(layout['creep'] & layout['failure'])
    for name in ('creep', 'failure'):
        assert np.count_nonzero(layout[name]) >= 100, name  # above coherence size


def test_comparison_control_area_avoids_the_creep_zone():
    """The report's control area must be stable ground, not the creep zone."""
    layout = showcase_layout()
    control = control_area(layout['mask'], layout['failure'])
    assert np.all(layout['zones'][control] == SHOWCASE_ZONES['stable'])


def test_simulated_motion_per_zone_without_noise():
    """Constant rates in stable and creep zones; failure accelerates after onset."""
    simulation = simulate_showcase(
        height=30,
        width=40,
        n_calibration=40,
        n_monitoring=20,
        noise_mm=0.0,
        atmosphere_mm=0.0,
        aps_amplitude_mm=0.0,
    )
    layout, series = simulation['layout'], simulation['series']
    velocity = np.diff(series, axis=0) / np.diff(simulation['offsets'])[:, None, None]
    stable = layout['mask'] & ~layout['creep'] & ~layout['failure']
    np.testing.assert_allclose(velocity[:, stable], -0.02)
    np.testing.assert_allclose(velocity[:, layout['creep']], -0.25)
    onset = simulation['dates'].index(simulation['failure_onset_date'])
    failure = velocity[:, layout['failure']].mean(axis=1)
    np.testing.assert_allclose(failure[:onset], -0.02)
    # After the onset: away from the satellite and faster at every acquisition.
    assert np.all(np.diff(failure[onset:]) < 0)
    assert failure[-1] < 2 * failure[onset]  # more than doubled by the end


def test_quick_site_is_a_valid_contract_with_enough_calibration(demo_run):
    config = load_config(demo_run / 'ale_showcase.yaml')
    contract = load_input(config)
    info = summary(contract)
    assert info['model_version'] == 'slope_trf@showcase'
    attrs = json.loads((demo_run / 'runs' / 'three_level' / 'alerts.json').read_text())[
        'engine_attrs'
    ]
    assert (
        attrs['n_calibration']
        >= config['engine_configs']['three_level']['calibration']['minimum_dates']
    )


def test_script_writes_runs_summary_and_report(demo_run):
    for path in (
        'runs/fixed_gates/alerts.npz',
        'runs/three_level/alerts.npz',
        'zone_summary.md',
        'report/report.md',
        'static/sectors.geojson',
    ):
        assert (demo_run / path).is_file(), path
    assert not (demo_run / 'plots').exists()
    text = (demo_run / 'zone_summary.md').read_text()
    for zone in ('stable', 'creep', 'failure'):
        assert f'| fixed_gates | {zone} |' in text
        assert f'| three_level | {zone} |' in text
    sectors = (demo_run / 'runs' / 'fixed_gates' / 'sectors.csv').read_text()
    for sector in ('west_beach', 'east_beach', 'dam_wall'):
        assert sector in sectors


@pytest.mark.parametrize(
    ('option', 'engine'), [('A', 'fixed_gates'), ('B', 'three_level')]
)
def test_script_runs_one_engine(demo_run, option, engine):
    assert (
        _script().main(['--out', str(demo_run), '--engine', option, '--no-plots']) == 0
    )
    assert {p.name for p in (demo_run / 'runs').iterdir()} == {engine, 'run.json'}
    assert not (demo_run / 'report').exists()  # earlier report removed


@pytest.mark.parametrize(
    ('alias', 'engine'), [('A', 'fixed_gates'), ('B', 'three_level')]
)
def test_run_accepts_engine_aliases(demo_run, tmp_path, alias, engine):
    written = run(alias, tmp_path / 'out', config_path=demo_run / 'ale_showcase.yaml')
    assert list(written) == [engine]


def test_script_refuses_the_repository_and_foreign_directories(tmp_path):
    script = _script()
    with pytest.raises(SystemExit):
        script.main(['--out', str(script.REPOSITORY / 'showcase_tmp')])
    (tmp_path / 'other.txt').write_text('not a showcase')
    with pytest.raises(SystemExit):
        script.main(['--out', str(tmp_path), '--quick'])
    assert (tmp_path / 'other.txt').is_file()


def test_zone_timeline_plot(demo_run, tmp_path):
    script = _script()
    import rasterio

    with rasterio.open(demo_run / 'static' / 'demo_zones.tif') as raster:
        zones = raster.read(1)
    scenario = json.loads((demo_run / 'scenario.json').read_text())
    levels = script.zone_levels(demo_run / 'runs', zones, ('three_level',))
    path = script.zone_timeline(
        tmp_path / 'zone_timeline.png',
        demo_run / 'results' / 'forecasts' / 'slope_trf' / 'residuals.npz',
        zones,
        levels,
        scenario,
    )
    assert path.stat().st_size > 10_000
    dates, means = script.zone_displacement(
        demo_run / 'results' / 'forecasts' / 'slope_trf' / 'residuals.npz', zones
    )
    # The failure zone ends far below the stable zone.
    assert means['failure'][-1] < means['stable'][-1] - 50
    assert len(dates) == len(means['creep'])
