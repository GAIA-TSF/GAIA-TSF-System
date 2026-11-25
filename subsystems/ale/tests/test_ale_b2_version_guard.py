"""B2.3: frozen calibrations are tied to (model_id, model_version, lead)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ale_3levels.__main__ import main
from ale_3levels.engines.three_level.calibration import (
    CalibrationMismatchError,
    MissingCalibrationError,
)
from ale_3levels.outputs import read_alerts
from ale_3levels.run import run
from ale_fixtures import write_long_table, write_run_config


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp('guard')
    v1 = write_long_table(root / 'v1', model_version='slope_trf@v1')
    v2 = write_long_table(root / 'v2', model_version='slope_trf@v2')
    config = write_run_config(root, v1)
    run('three_level', root / 'first', config_path=config)
    calibration = root / 'first' / 'three_level' / 'engine_state' / 'calibration'
    return root, v1, v2, config, calibration


def test_calibration_records_its_model_identity(setup):
    *_, calibration = setup
    for level in ('l1', 'l2', 'l3'):
        path = calibration / 'trf' / 'lead1' / f'{level}_calibration.json'
        payload = json.loads(path.read_text())
        assert payload['identity'] == {
            'model_id': 'trf',
            'model_version': 'slope_trf@v1',
            'lead': 1,
        }


def test_other_model_version_is_refused(setup):
    root, _, v2, config, calibration = setup
    with pytest.raises(CalibrationMismatchError) as error:
        run(
            'three_level',
            root / 'v2_run',
            config_path=config,
            input_path=v2,
            calibration_dir=calibration,
        )
    message = str(error.value)
    assert 'slope_trf@v1' in message and 'slope_trf@v2' in message


def test_cli_reports_the_mismatch(setup, capsys):
    root, _, v2, config, calibration = setup
    code = main(
        [
            'run',
            '--engine',
            'three_level',
            '--config',
            str(config),
            '--input',
            str(v2),
            '--out',
            str(root / 'cli'),
            '--calibration',
            str(calibration),
        ]
    )
    assert code == 1
    assert 'Calibration error' in capsys.readouterr().err


def test_same_version_reuses_the_frozen_calibration(setup):
    root, _, _, config, calibration = setup
    run('three_level', root / 'again', config_path=config, calibration_dir=calibration)
    first, _ = read_alerts(root / 'first' / 'three_level')
    again, meta = read_alerts(root / 'again' / 'three_level')
    assert meta['engine_attrs']['calibration_reused'] == ['l1', 'l2', 'l3']
    np.testing.assert_array_equal(again['level'], first['level'])
    np.testing.assert_array_equal(again['metric_z'], first['metric_z'])
    # No new calibration was written for the reusing run.
    assert not (
        root / 'again' / 'three_level' / 'engine_state' / 'calibration'
    ).exists()


def test_missing_frozen_calibration_is_reported(setup, tmp_path):
    _, _, _, config, _ = setup
    with pytest.raises(MissingCalibrationError, match="Run 'calibrate' first"):
        run(
            'three_level',
            tmp_path / 'out',
            config_path=config,
            calibration_dir=tmp_path / 'nothing',
        )
