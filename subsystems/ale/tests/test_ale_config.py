"""Configuration layering (base files) and relative path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from ale_3levels.config import DEFAULT_CONFIG, load_config

CONFIG_DIR = DEFAULT_CONFIG.parent


def test_site_config_extends_the_default(tmp_path):
    config = load_config(CONFIG_DIR / 'ale_synthetic.yaml')
    default = load_config()
    assert config['input']['source_units_override'] == 'm'
    assert config['input']['lead'] == default['input']['lead']
    assert config['spatial'] == default['spatial']
    assert config['engine_configs'] == default['engine_configs']
    assert 'base' not in config


def test_paths_expand_home_and_stay_outside_the_repository():
    config = load_config(CONFIG_DIR / 'ale_synthetic.yaml')
    residuals = Path(config['input']['residuals'])
    assert residuals.is_absolute()
    assert residuals == (
        Path.home()
        / 'ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV/synthetic_site'
        / 'results/forecasts/slope_trf/residuals.npz'
    )
    repository = CONFIG_DIR.parents[2]
    assert repository not in residuals.parents


def test_relative_paths_resolve_against_the_file_that_sets_them(tmp_path):
    (tmp_path / 'site.yaml').write_text(
        f'base: {CONFIG_DIR / "ale.yaml"}\ninput:\n  residuals: data/residuals.npz\n'
    )
    config = load_config(tmp_path / 'site.yaml')
    assert Path(config['input']['residuals']) == tmp_path / 'data' / 'residuals.npz'
    # Engine files named in the base resolve against the base's directory.
    assert config['engine_configs'] == load_config()['engine_configs']


def test_circular_base_is_rejected(tmp_path):
    (tmp_path / 'a.yaml').write_text('base: b.yaml\n')
    (tmp_path / 'b.yaml').write_text('base: a.yaml\n')
    with pytest.raises(ValueError, match='Circular'):
        load_config(tmp_path / 'a.yaml')
