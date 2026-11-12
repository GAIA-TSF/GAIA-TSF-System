"""Tests for LSTM validation-loss early stopping."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('torch')

from subsystems.map.plugins.models.lstm import LSTMModel  # noqa: E402


def _config(**early_stopping) -> dict:
    return {
        'look_back': 3,
        'horizon': 1,
        'hidden_size': 8,
        'num_layers': 1,
        'dropout': 0.0,
        'device': 'cpu',
        'optimizer': {'lr': 0.05},
        'training': {
            'batch_size': 32,
            'epochs': 60,
            'early_stopping': {'enabled': True, 'patience': 3, **early_stopping},
        },
    }


def _data(seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(size, 3, 2))
    targets = features[:, -1, 0] + 0.1 * rng.normal(size=size)
    return features, targets


def _trained(**early_stopping) -> LSTMModel:
    model = LSTMModel(_config(**early_stopping))
    model.set_random_seed(1)
    # Validation targets unrelated to the inputs: the loss stops improving
    # early, so training must stop well before the 60-epoch limit.
    validation_features, _ = _data(2, 64)
    model.set_validation_data(
        validation_features, np.random.default_rng(3).normal(size=64)
    )
    model.train(*_data(1, 256))
    return model


def test_early_stopping_stops_and_restores_best_epoch() -> None:
    model = _trained()

    history = model.validation_history
    assert model.stopped_epoch is not None and model.stopped_epoch < 60
    assert len(history) == model.stopped_epoch
    assert model.best_epoch == int(np.argmin(history)) + 1
    assert model.stopped_epoch - model.best_epoch == 3
    # Restored weights reproduce the best epoch's validation loss.
    criterion = model._nn.MSELoss()
    restored = model._validation_loss(
        criterion, model._normalized_validation_tensors()
    )
    assert restored == pytest.approx(min(history), rel=1e-5)


def test_training_summary_and_persistence(tmp_path) -> None:
    model = _trained()
    path = tmp_path / 'model.pkl'
    model.save(path)

    loaded = LSTMModel.load(path)
    summary = loaded.training_summary()

    assert summary['best_epoch'] == model.best_epoch
    assert summary['epochs_run'] == model.stopped_epoch
    assert summary['best_validation_loss'] == pytest.approx(
        min(model.validation_history)
    )
    features, _ = _data(4, 8)
    np.testing.assert_allclose(
        loaded.predict(features).y_pred, model.predict(features).y_pred, rtol=1e-6
    )


def test_disabled_early_stopping_runs_all_epochs() -> None:
    config = _config()
    config['training']['epochs'] = 4
    config['training']['early_stopping'] = {'enabled': False}
    model = LSTMModel(config)
    model.set_random_seed(1)
    model.train(*_data(1, 64))

    assert len(model.training_history) == 4
    assert model.best_epoch is None and model.stopped_epoch is None


def test_invalid_patience_is_rejected() -> None:
    with pytest.raises(ValueError, match='patience'):
        LSTMModel(_config(patience=0))
