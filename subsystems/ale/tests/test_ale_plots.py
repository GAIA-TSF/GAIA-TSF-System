"""Plots read saved results only; enabling them changes no number (spec §5.1/§6.1)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml
from ale_3levels.__main__ import main
from ale_3levels.demo import write_demo_site
from ale_3levels.run import run

PLOTS = Path(__file__).resolve().parents[1] / 'ale_3levels' / 'plots.py'


def _numerical_files(directory):
    """Every output except plots and run.json; alerts.json without its config path.

    The plots run uses a second config file (outputs.plots on), so only the
    recorded config path may differ.
    """
    files = {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or 'plots' in path.parts or path.name == 'run.json':
            continue
        key = str(path.relative_to(directory))
        if path.name == 'alerts.json':
            payload = json.loads(path.read_text())
            payload.pop('ale_config_path')
            files[key] = payload
        else:
            files[key] = path.read_bytes()
    return files


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp('plots')
    demo = write_demo_site(root / 'site', size=16, n_dates=40)
    run('both', root / 'off', config_path=demo['config'])
    raw = yaml.safe_load(demo['config'].read_text())
    raw['outputs'] = {'plots': True, 'animations': True}
    raw['input'] = {
        **raw['input'],
        'residuals': str(demo['table']),
        'mask_file': str(demo['tsf_mask']),
    }
    config = root / 'site' / 'ale_plots.yaml'
    config.write_text(yaml.safe_dump(raw))
    run('both', root / 'on', config_path=config)
    return root


def test_plots_do_not_change_numerical_outputs(runs):
    assert _numerical_files(runs / 'on') == _numerical_files(runs / 'off')


def test_plots_and_animation_are_written(runs):
    plots = runs / 'on' / 'plots'
    for name in (
        'three_level_history.png',
        'sector_timeline.png',
        'fixed_gates_level_counts.png',
        'three_level_level_counts.png',
        'alerts.gif',
    ):
        assert (plots / name).stat().st_size > 0, name
    assert len(list((plots / 'maps').glob('alerts_*.png'))) > 0
    assert not (runs / 'off' / 'plots').exists()


def test_plot_command_on_a_saved_run(runs, capsys):
    assert (
        main(
            [
                'plot',
                '--run',
                str(runs / 'off'),
                '--out',
                str(runs / 'replot'),
                '--no-maps',
            ]
        )
        == 0
    )
    assert (runs / 'replot' / 'three_level_history.png').is_file()


def test_plots_use_only_saved_outputs():
    tree = ast.parse(PLOTS.read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    ale = {module for module in imported if module.startswith('ale_3levels')}
    assert ale == {'ale_3levels.outputs'}
