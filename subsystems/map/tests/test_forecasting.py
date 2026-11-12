"""Forecasting contract of MAP for both key variables and all three models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import DATES, MAP_CONFIG, project_config, slope_stacks, write_project
from subsystems.map.dataset import LoadedFeatures, RasterGrid, build_forecast_samples, split_temporal
from subsystems.map.pipelines.inference_pipeline import run_inference
from subsystems.map.pipelines.learning_pipeline import predict_values, run_learning
from subsystems.map.plugins.models.const import ConstantBaselineModel
from subsystems.map.utils.config_loader import MapContext, load_config
from subsystems.map.utils.forecasts import compare_residual_tables, read_table


ALERT_WORDS = ('alert', 'anomal', 'cusum', 'zscore', 'z_score', 'level1', 'level2',
               'level3', 'regime', 'persistent', 'coheren', 'dashboard', 'exceed',
               'threshold')


def _loaded(target: np.ndarray, features: dict[str, np.ndarray] | None = None) -> LoadedFeatures:
    times = target.shape[0]
    dates = tuple(DATES[:times])
    stacks = {'target': target, 'x': np.zeros_like(target), **(features or {})}
    return LoadedFeatures(
        stacks, dates, RasterGrid(None, (1, 0, 0, 0, -1, 0), 1, target.shape[2], None),
        np.ones(target.shape[1:], dtype=bool),
    )


def _run(root: Path, dataset: str, model: str, horizon: int = 3) -> tuple[dict, dict]:
    config = project_config(root, dataset, model, horizon)
    return run_learning(config), run_inference(config)


def _forecasts(root: Path, experiment: str) -> dict[str, np.ndarray]:
    return read_table(root / 'results' / 'forecasts' / experiment / 'forecasts')


# --- persistence --------------------------------------------------------------

def test_constant_repeats_last_valid_value_for_every_lead() -> None:
    target = np.array([1.0, 4.0, 7.0, np.nan, 9.0, 9.5])[:, None, None]
    samples = build_forecast_samples(_loaded(target), ['x'], 'target', horizon=3)
    model = ConstantBaselineModel({}, horizon=3)
    predicted = predict_values(model, samples)
    at_seven = samples.origin_indices == 2
    np.testing.assert_array_equal(predicted[at_seven], [[7.0, 7.0, 7.0]])
    # Origin 3 has no observation: the last valid value at t is still 7.
    np.testing.assert_array_equal(predicted[samples.origin_indices == 3], [[7.0, 7.0, 7.0]])


# --- forecast convention ------------------------------------------------------

@pytest.mark.parametrize('model', ['trf', 'lstm'])
def test_trf_and_lstm_origin_target_dates_and_leads(slope_project: Path, model: str) -> None:
    if model == 'lstm':
        pytest.importorskip('torch')
    _run(slope_project, 'slope_dataset', model)
    table = _forecasts(slope_project, f'slope_{model}')
    position = {value: index for index, value in enumerate(DATES)}
    origin = np.array([position[value] for value in table['forecast_origin_date']])
    assert set(table['lead']) == {1, 2, 3}
    acquired = table['target_date_status'] == 'acquired'
    expected_dates = np.array(DATES)[np.minimum(origin + table['lead'], len(DATES) - 1)]
    np.testing.assert_array_equal(table['target_date'][acquired], expected_dates[acquired])
    # Beyond the last acquisition, target dates are projected at 12 days.
    last = np.datetime64(DATES[-1])
    beyond = origin + table['lead'] - (len(DATES) - 1)
    projected = (last + (beyond[~acquired] * 12).astype('timedelta64[D]')).astype(str)
    np.testing.assert_array_equal(table['target_date'][~acquired], projected)
    assert max(table['forecast_origin_date']) == DATES[-1]
    # acceleration_lag3 first exists at t=5; the LSTM window (look_back 3)
    # ends at the origin, so its first origin is two acquisitions later.
    assert origin.min() == (5 if model == 'trf' else 7)
    assert np.all(np.isfinite(table['predicted_value']))


@pytest.mark.parametrize('model', ['constant', 'trf', 'lstm'])
def test_future_observations_do_not_change_issued_forecasts(
    tmp_path: Path, model: str
) -> None:
    if model == 'lstm':
        pytest.importorskip('torch')
    root = write_project(tmp_path / 'p', 'slope')
    config = project_config(root, 'slope_dataset', model)
    run_learning(config)
    run_inference(config)
    before = _forecasts(root, f'slope_{model}')
    cutoff = 30
    changed = slope_stacks()
    for name, values in changed.items():
        values[cutoff + 1 :] = values[cutoff + 1 :] * 3.0 - 50.0
    write_project(root, 'slope', changed)
    run_inference(config)
    after = _forecasts(root, f'slope_{model}')
    issued = before['forecast_origin_date'] <= DATES[cutoff]
    issued_after = after['forecast_origin_date'] <= DATES[cutoff]
    for field in ('forecast_origin_date', 'lead', 'series_id', 'target_date', 'predicted_value'):
        np.testing.assert_array_equal(before[field][issued], after[field][issued_after])
    assert not np.allclose(
        before['predicted_value'][~issued], after['predicted_value'][~issued_after]
    )


def test_temporal_split_purges_targets_past_each_subset() -> None:
    target = np.arange(20, dtype=float)[:, None, None]
    samples = build_forecast_samples(_loaded(target), ['x'], 'target', horizon=3)
    splits = split_temporal(samples, 0, 20, 0.6, 0.2, 0.2)
    for subset, end in ((splits.train, 12), (splits.validation, 16), (splits.test, 20)):
        index = subset.origin_indices[:, None] + np.arange(1, 4)
        assert np.all(np.isnan(subset.targets[index >= end]))
        assert np.all(np.isfinite(subset.targets[index < end]))


def test_lagged_target_is_rejected() -> None:
    config = load_config(MAP_CONFIG)
    config['datasets']['slope_dataset']['target_feature'] = 'velocity_lag1'
    with pytest.raises(ValueError, match='lagged'):
        MapContext.from_config(config)


# --- residuals ----------------------------------------------------------------

def test_slope_residual_sign_and_units(slope_project: Path) -> None:
    _run(slope_project, 'slope_dataset', 'constant')
    output = slope_project / 'results' / 'forecasts' / 'slope_constant'
    residuals = read_table(output / 'residuals')
    sidecar = json.loads((output / 'residuals.json').read_text())
    assert sidecar['target_variable'] == 'cumulative_los'
    assert sidecar['target_units'] == 'mm'
    assert sidecar['fields']['residual'] == 'mm'
    np.testing.assert_allclose(
        residuals['residual'], residuals['observed_value'] - residuals['predicted_value']
    )
    # LOS moves away from the satellite (decreasing cumulative mm), so the
    # observation falls below persistence: negative residuals, not flipped.
    assert np.mean(residuals['residual'] < 0) > 0.9
    # Observed values are the cumulative LOS input at the target date.
    stack = slope_stacks()['cumulative_los_current']
    position = {value: index for index, value in enumerate(DATES)}
    time = np.array([position[value] for value in residuals['target_date']])
    np.testing.assert_allclose(
        residuals['observed_value'], stack[time, residuals['row'], residuals['col']], rtol=1e-6
    )


def test_missing_acquisition_does_not_shift_matches(slope_project: Path) -> None:
    _run(slope_project, 'slope_dataset', 'constant')
    output = slope_project / 'results' / 'forecasts' / 'slope_constant'
    forecasts, residuals = read_table(output / 'forecasts'), read_table(output / 'residuals')
    pixel = 1 * 5 + 1  # cumulative LOS at DATES[17] is missing for this pixel
    in_pixel = residuals['series_id'] == pixel
    assert DATES[17] not in set(residuals['target_date'][in_pixel])
    for date in (DATES[16], DATES[18]):
        assert np.any(residuals['target_date'][in_pixel] == date)
    # Forecasts for the missing date are saved even without the observation.
    issued = (forecasts['series_id'] == pixel) & (forecasts['target_date'] == DATES[17])
    assert np.any(issued)


def test_amd_runs_with_native_units_and_cloud_gaps(amd_project: Path) -> None:
    for model in ('constant', 'trf'):
        learning, inference = _run(amd_project, 'amd_dataset', model)
        assert learning['kv'] == 'amd' and learning['target_variable'] == 'amd_difference'
        assert inference['matched_rows'] > 0
    output = amd_project / 'results' / 'forecasts' / 'amd_trf'
    residuals = read_table(output / 'residuals')
    assert json.loads((output / 'residuals.json').read_text())['fields']['residual'] == 'unconfirmed'
    np.testing.assert_allclose(
        residuals['residual'], residuals['observed_value'] - residuals['predicted_value']
    )
    pixel = 2 * 5 + 3  # AMD missing at DATES[22] for this pixel
    assert DATES[22] not in set(residuals['target_date'][residuals['series_id'] == pixel])


@pytest.mark.parametrize('horizon', [1, 5])
def test_amd_lstm_supports_configured_horizon(amd_project: Path, horizon: int) -> None:
    pytest.importorskip('torch')
    config = project_config(amd_project, 'amd_dataset', 'lstm', horizon)
    config['datasets']['amd_dataset']['features'] = ['annual_sin', 'annual_cos']
    run_learning(config)
    run_inference(config)
    assert set(_forecasts(amd_project, 'amd_lstm')['lead']) == set(range(1, horizon + 1))


# --- validation and handoff products -------------------------------------------

def test_validation_metrics_compare_models_with_constant_on_same_rows(slope_project: Path) -> None:
    learning, _ = _run(slope_project, 'slope_dataset', 'trf')
    _run(slope_project, 'slope_dataset', 'constant')
    for row in learning['metrics']['validation']:
        assert {'mae', 'rmse', 'mean_signed_error', 'valid_count', 'constant_mae',
                'constant_rmse', 'constant_mean_signed_error', 'interval_coverage'} <= set(row)
    csv_text = (slope_project / 'results/models/slope_trf/validation_metrics.csv').read_text()
    assert csv_text.splitlines()[0].startswith('kv,target_variable,target_units,model')
    base = slope_project / 'results' / 'forecasts'
    tables = {
        'trf': read_table(base / 'slope_trf' / 'residuals'),
        'constant_model': read_table(base / 'slope_constant' / 'residuals'),
    }
    comparison = compare_residual_tables(tables, roles=('validation', 'test'))
    assert [row['lead'] for row in comparison] == [1, 2, 3]
    for row in comparison:
        assert row['trf']['valid_count'] == row['constant']['valid_count'] > 0
        # The constant model is persistence, so it equals the shared baseline.
        assert row['constant_model'] == row['constant']


def test_outputs_exist_without_plots_and_without_alert_products(slope_project: Path) -> None:
    _, inference = _run(slope_project, 'slope_dataset', 'trf')
    files = [path for path in (slope_project / 'results').rglob('*') if path.is_file()]
    produced = [path for path in files if 'models' in path.parts or 'forecasts' in path.parts]
    names = {path.name for path in produced}
    assert {'model.pkl', 'experiment.json', 'validation_metrics.csv', 'reference_forecasts.npz',
            'forecasts.npz', 'forecasts.json', 'residuals.npz', 'residuals.json',
            'forecast_errors.csv', 'inference.json'} <= names
    assert not [path for path in produced if path.suffix in {'.png', '.mp4', '.gif'}]
    for path in produced:
        assert not any(word in path.name.lower() for word in ALERT_WORDS), path
    sidecar = json.loads((slope_project / 'results/forecasts/slope_trf/forecasts.json').read_text())
    assert not any(word in key for key in sidecar['fields'] for word in ALERT_WORDS)
    assert {'model_version', 'grid', 'interval_level'} <= set(sidecar)
    roles = read_table(slope_project / 'results/forecasts/slope_trf/residuals')['sample_role']
    assert {'validation', 'test'} <= set(roles)  # held-out normal errors for ALE
    assert inference['expected_target_rows'] > 0


def test_shipped_config_has_no_alert_settings() -> None:
    text = MAP_CONFIG.read_text().lower()
    config = yaml.safe_load(text)
    assert not set(config) & {'anomaly_detection', 'monitoring', 'baseline_model'}
    for word in ('cusum', 'zscore', 'regime', 'dashboard', 'anomal', 'animation'):
        assert word not in text
    config = load_config(MAP_CONFIG)
    config['anomaly_detection'] = {'zscore_threshold': 1.5}
    with pytest.raises(KeyError, match='anomaly_detection'):
        MapContext.from_config(config)
