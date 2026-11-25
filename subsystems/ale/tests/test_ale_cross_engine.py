"""Cross-engine properties (spec §5.1): no look-ahead and threshold replay.

Determinism (byte-identical reruns) is tested in test_ale_run_outputs.py, the
import boundary in test_ale_import_boundary.py, and no look-ahead with a
persisted engine-B state in test_ale_b2_state.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import yaml
from ale_3levels.config import load_config
from ale_3levels.demo import write_demo_site
from ale_3levels.outputs import read_alerts
from ale_3levels.run import run

ENGINES = ('fixed_gates', 'three_level')


@pytest.fixture(scope='module')
def sites(tmp_path_factory):
    root = tmp_path_factory.mktemp('cross')
    # Same simulation; the full site has 10 later acquisitions appended.
    prefix = write_demo_site(root / 'prefix', n_dates=50)
    full = write_demo_site(root / 'full', n_dates=60)
    run('both', root / 'out_prefix', config_path=prefix['config'])
    run('both', root / 'out_full', config_path=full['config'])
    return root, prefix, full


def _digest(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize('engine', ENGINES)
def test_appending_acquisitions_does_not_change_earlier_results(sites, engine):
    root, _, _ = sites
    earlier, _ = read_alerts(root / 'out_prefix' / engine)
    later, _ = read_alerts(root / 'out_full' / engine)
    n = earlier['level'].shape[0]
    assert later['level'].shape[0] == n + 10
    np.testing.assert_array_equal(later['dates'][:n], earlier['dates'])
    for key, values in earlier.items():
        if key in ('dates', 'mask'):
            continue
        np.testing.assert_array_equal(later[key][:n], values, err_msg=key)


def _replay_config(root, base_config, engine, section_updates):
    """A copy of the site config with one engine's thresholds changed."""
    engines = load_config()['engine_configs']
    changed = dict(engines[engine])
    for key, value in section_updates.items():
        changed[key] = {**changed[key], **value} if isinstance(value, dict) else value
    directory = root / f'replay_{engine}'
    directory.mkdir(exist_ok=True)
    for name in ENGINES:
        payload = changed if name == engine else engines[name]
        (directory / f'{name}.yaml').write_text(yaml.safe_dump(payload))
    raw = yaml.safe_load(base_config.read_text())
    site_dir = base_config.parent
    raw['input']['residuals'] = str(site_dir / raw['input']['residuals'])
    raw['input']['mask_file'] = str(site_dir / raw['input']['mask_file'])
    raw['engines'] = {name: f'{name}.yaml' for name in ENGINES}
    path = directory / 'ale.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    ('engine', 'updates'),
    [
        ('fixed_gates', {'gates': {'white_below': 2.0, 'green_below': 6.0}}),
        ('three_level', {'level1': {'zscore_threshold': 2.5}}),
    ],
)
def test_threshold_replay_without_touching_map(sites, engine, updates):
    root, _, full = sites
    map_outputs = full['table'].parent
    before = _digest(map_outputs)
    config = _replay_config(root, full['config'], engine, updates)
    run('both', root / f'out_replay_{engine}', config_path=config)
    assert _digest(map_outputs) == before  # MAP outputs neither rerun nor modified
    for name in ENGINES:
        original, meta_original = read_alerts(root / 'out_full' / name)
        replay, meta_replay = read_alerts(root / f'out_replay_{engine}' / name)
        if name == engine:
            assert meta_replay['config_hash'] != meta_original['config_hash']
            assert not np.array_equal(replay['level'], original['level'])
        else:
            assert meta_replay['config_hash'] == meta_original['config_hash']
            np.testing.assert_array_equal(replay['level'], original['level'])
