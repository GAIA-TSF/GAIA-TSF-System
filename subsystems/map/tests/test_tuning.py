"""Hyperparameter tuning by rolling-origin cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import project_config
from subsystems.map.dataset import build_forecast_samples, rolling_origin_folds
from subsystems.map.pipelines.tuning_pipeline import apply_params, candidate_grid, run_tuning
from test_forecasting import _loaded


def test_rolling_origin_folds_expand_and_purge() -> None:
    target = np.arange(24, dtype=float)[:, None, None]
    samples = build_forecast_samples(_loaded(target), ['x'], 'target', horizon=2)
    folds = rolling_origin_folds(samples, 3)
    previous_end = None
    for fit, validation in folds:
        fit_end = int(validation.origin_indices.min())
        assert fit.origin_indices.max() < fit_end
        assert fit.origin_indices.min() == samples.origin_indices.min()  # expanding
        lead_index = fit.origin_indices[:, None] + np.arange(1, 3)
        assert np.all(np.isnan(fit.targets[lead_index >= fit_end]))
        assert previous_end is None or fit_end > previous_end
        previous_end = fit_end


def test_candidate_grid_and_dotted_params() -> None:
    grid = candidate_grid({'a': [1, 2], 'b': [3, 4, 5]}, limit=10, seed=0)
    assert len(grid) == 6
    assert candidate_grid({'a': [1, 2], 'b': [3, 4, 5]}, 4, 0) == candidate_grid(
        {'a': [1, 2], 'b': [3, 4, 5]}, 4, 0
    )
    merged = apply_params({'optimizer': {'lr': 0.1}, 'look_back': 3}, {'optimizer.lr': 0.5})
    assert merged == {'optimizer': {'lr': 0.5}, 'look_back': 3}


@pytest.mark.parametrize('model', ['trf', 'lstm'])
def test_tuning_selects_best_and_retrains(slope_project: Path, model: str) -> None:
    if model == 'lstm':
        pytest.importorskip('torch')
    config = project_config(slope_project, 'slope_dataset', model)
    config['datasets']['slope_dataset']['split']['method'] = 'temporal'
    config['tuning'] = {
        'folds': 2,
        'max_candidates': 3,
        'search': {
            'trf': {'max_depth': [2, None], 'min_samples_leaf': [1, 10]},
            'lstm': {'look_back': [2, 3], 'hidden_size': [4]},
        },
    }
    summary = run_tuning(config)
    output = slope_project / 'results' / 'tuning' / f'slope_{model}'
    best = json.loads((output / 'best_params.json').read_text())
    assert best['candidate_count'] == (3 if model == 'trf' else 2)
    rows = (output / 'tuning_results.csv').read_text().splitlines()
    assert len(rows) - 1 == best['candidate_count'] * 2 * 3  # candidates x folds x leads
    tuned = yaml.safe_load((output / 'tuned_config.yaml').read_text())['models'][model]
    for key, value in best['best_params'].items():
        assert tuned[key] == value
    experiment = json.loads(
        (slope_project / 'results' / 'models' / f'slope_{model}' / 'experiment.json').read_text()
    )
    assert experiment['model_config'] == tuned  # retrained with the selected settings
    assert 'test' in summary['learning']
