"""B2.5: configuration selects the MAP model; each model has its own calibration."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from ale_3levels.__main__ import main
from ale_3levels.config import load_config
from ale_3levels.contract import ContractError
from ale_3levels.engines.three_level.calibration import MissingCalibrationError
from ale_3levels.outputs import read_alerts
from ale_3levels.run import load_input, run
from ale_fixtures import write_long_table, write_run_config


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp('models')
    trf = write_long_table(root / 'trf', model_version='slope_trf@1')
    lstm = write_long_table(
        root / 'lstm', model_id='lstm', model_version='slope_lstm@1', seed=11
    )
    config = write_run_config(root, trf)
    raw = yaml.safe_load(config.read_text())
    raw['input']['residuals'] = {'trf': str(trf), 'lstm': str(lstm)}
    config.write_text(yaml.safe_dump(raw))
    return root, config


@pytest.mark.parametrize('model', ['trf', 'lstm'])
def test_model_id_selects_the_table(setup, model):
    _, config = setup
    contract = load_input(load_config(config), model_id=model)
    assert contract.attrs['model_id'] == model


def test_default_model_comes_from_the_configuration(setup):
    _, config = setup
    assert load_input(load_config(config)).attrs['model_id'] == 'trf'


def test_each_model_has_its_own_calibration(setup):
    root, config = setup
    for model in ('trf', 'lstm'):
        assert (
            main(
                [
                    'run',
                    '--engine',
                    'three_level',
                    '--config',
                    str(config),
                    '--model',
                    model,
                    '--out',
                    str(root / model / 'out'),
                ]
            )
            == 0
        )
    trf, meta_trf = read_alerts(root / 'trf' / 'out' / 'three_level')
    lstm, meta_lstm = read_alerts(root / 'lstm' / 'out' / 'three_level')
    assert meta_trf['engine_attrs']['calibration_identity']['model_id'] == 'trf'
    assert meta_lstm['engine_attrs']['calibration_identity']['model_id'] == 'lstm'
    assert meta_lstm['input']['model_version'] == 'slope_lstm@1'
    assert not np.array_equal(trf['metric_z'], lstm['metric_z'], equal_nan=True)
    calibration = root / 'trf' / 'out' / 'three_level' / 'engine_state' / 'calibration'
    assert (calibration / 'trf' / 'lead1' / 'l1_calibration.json').is_file()
    # The trf calibration cannot serve the lstm residuals.
    with pytest.raises(MissingCalibrationError):
        run(
            'three_level',
            root / 'cross',
            config_path=config,
            model_id='lstm',
            calibration_dir=calibration,
        )


def test_unknown_model_is_rejected(setup):
    _, config = setup
    with pytest.raises(ValueError, match="no table for model 'gbr'"):
        load_input(load_config(config), model_id='gbr')


def test_model_must_match_the_table(setup):
    root, _ = setup
    config = write_run_config(root / 'single', root / 'lstm' / 'residuals.npz')
    with pytest.raises(ContractError, match="selects 'trf'"):
        load_input(load_config(config))
