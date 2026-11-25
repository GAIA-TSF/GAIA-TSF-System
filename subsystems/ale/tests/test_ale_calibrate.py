"""'calibrate' freezes engine-B calibration that 'run --calibration' uses."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ale_3levels.__main__ import main
from ale_3levels.demo import write_demo_site
from ale_3levels.outputs import read_alerts
from ale_3levels.run import calibrate, run


@pytest.fixture(scope='module')
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp('calibrate')
    return root, write_demo_site(root / 'site')


def test_calibrate_writes_frozen_files_and_manifest(site):
    root, demo = site
    calibration = calibrate(root / 'frozen', config_path=demo['config'])
    folder = calibration / 'trf' / 'lead1'
    assert sorted(path.name for path in folder.iterdir()) == [
        'l1_calibration.json',
        'l2_calibration.json',
        'l3_calibration.json',
    ]
    manifest = json.loads((root / 'frozen' / 'calibration_manifest.json').read_text())
    assert manifest['identity'] == {
        'model_id': 'trf',
        'model_version': 'slope_trf@demo',
        'lead': 1,
    }
    assert manifest['calibration_population'] == 'independent'
    with pytest.raises(FileExistsError):
        calibrate(root / 'frozen', config_path=demo['config'])


def test_run_with_frozen_calibration_matches_a_fresh_run(site):
    root, demo = site
    calibration = root / 'frozen' / 'calibration'
    if not calibration.exists():
        calibrate(root / 'frozen', config_path=demo['config'])
    assert (
        main(
            [
                'run',
                '--engine',
                'three_level',
                '--config',
                str(demo['config']),
                '--out',
                str(root / 'with'),
                '--calibration',
                str(calibration),
            ]
        )
        == 0
    )
    run('three_level', root / 'fresh', config_path=demo['config'])
    frozen, meta = read_alerts(root / 'with' / 'three_level')
    fresh, _ = read_alerts(root / 'fresh' / 'three_level')
    assert meta['engine_attrs']['calibration_reused'] == ['l1', 'l2', 'l3']
    for key in fresh:
        np.testing.assert_array_equal(frozen[key], fresh[key], err_msg=key)


def test_cli_calibrate(site, capsys):
    root, demo = site
    assert (
        main(['calibrate', '--config', str(demo['config']), '--out', str(root / 'cli')])
        == 0
    )
    assert capsys.readouterr().out.strip().endswith('calibration')
