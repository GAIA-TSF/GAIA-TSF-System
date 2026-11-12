"""Tests for MAP contract v1.0 clauses: predictors, run mode, quality, exports."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from subsystems.dag.utils.missing_values import handle_missing_values
from subsystems.dag.utils.normalization import normalize_features
from subsystems.dag.utils.outliers import transform_outliers
from subsystems.map.dataset.dataset_builder import validate_predictors
from subsystems.map.monitoring.quality import (
    QualityParameters,
    acquisition_quality,
    run_quality_flags,
)
from subsystems.map.monitoring.series_export import build_records, write_series
from subsystems.map.utils.contract import (
    compatible_schema,
    file_sha256,
    training_run_mode,
    verify_training_inputs,
)
from subsystems.map.utils.forecast_timing import forecast_timing


def test_predictors_must_be_lagged_or_calendar() -> None:
    validate_predictors(['velocity_lag2', 'acceleration_lag1', 'annual_sin'])
    for unsafe in ('velocity_roll_mean', 'velocity_current', 'velocity', 'x_lag0'):
        with pytest.raises(ValueError, match=unsafe):
            validate_predictors(['velocity_lag2', unsafe])


def _experiment(method: str, end: str) -> dict:
    return {
        'calibration_spatial_selection': {
            'method': method, 'selection_end_date': end,
        },
        'split_date_ranges': {'train': {'end_date': end}},
    }


def test_run_mode_follows_contract_rule() -> None:
    # Stable polygon: operational even when training spans monitoring dates.
    assert training_run_mode(_experiment('configured_mask', '2020-12-23'), '2018-01-01')[0] == 'operational'
    assert training_run_mode(_experiment('temporal_std', '2017-12-27'), '2018-01-01')[0] == 'operational'
    assert training_run_mode(_experiment('temporal_std', '2020-12-23'), '2018-01-01')[0] == 'retrospective'
    assert training_run_mode(
        _experiment('configured_mask_and_temporal_std', '2020-12-23'), '2018-01-01'
    )[0] == 'retrospective'
    assert training_run_mode({}, '2018-01-01')[0] == 'unknown'


def test_schema_versions_and_stable_mask_fingerprint(tmp_path) -> None:
    assert compatible_schema('1.0') and compatible_schema('1.0.0')
    assert not compatible_schema('2.0')
    mask = tmp_path / 'stable.tif'
    mask.write_bytes(b'a')
    experiment = {
        'calibration_spatial_selection': {
            'mask_file': str(mask), 'mask_sha256': file_sha256(mask),
        }
    }
    verify_training_inputs(experiment)
    mask.write_bytes(b'b')
    with pytest.raises(ValueError, match='changed since training'):
        verify_training_inputs(experiment)


def test_forecast_timing_reports_lead_time() -> None:
    dataset = SimpleNamespace(
        dates=('2020-01-01', '2020-01-13', '2020-01-25'),
        forecast_origin_dates=(None, None, '2020-01-01'),
        source_dates=(),
        target_feature='velocity_current',
    )

    timing = forecast_timing(dataset)

    assert timing['schema_version'] == '1.0'
    assert timing['series'][2]['lead_time_days'] == 24
    assert timing['series'][2]['lead_time_acquisitions'] == 2
    assert timing['lead_time'] == {
        'acquisitions': {'min': 2, 'max': 2}, 'days': {'min': 24, 'max': 24},
    }


def test_quality_states_follow_contract_order() -> None:
    mask = np.ones((2, 2), dtype=bool)
    observed = np.ones((5, 2, 2))
    predicted = np.ones((5, 2, 2))
    predicted[1] = np.nan                 # warm_up
    observed[2] = np.nan                  # missing_acquisition
    observed[3, 0, :] = np.nan
    observed[3, 1, 0] = np.nan            # 25 % coverage

    states, coverage = acquisition_quality(
        observed, predicted, mask, (1, 5), QualityParameters()
    )

    assert states.tolist() == [
        'not_monitored', 'warm_up', 'missing_acquisition',
        'insufficient_coverage', 'ok',
    ]
    assert coverage[3] == pytest.approx(0.25)


def test_short_calibration_is_flagged_and_warned(caplog) -> None:
    flags = run_quality_flags(
        run_mode='retrospective',
        l2_bias_warning=False,
        calibration_dates={'l1': 14, 'l2': 14, 'l3': 30},
        calibrations_reused=[],
        parameters=QualityParameters(),
    )

    names = [flag['flag'] for flag in flags]
    assert names == ['retrospective_training', 'short_calibration']
    assert flags[1]['calibration_dates'] == {'l1': 14, 'l2': 14}
    assert 'short_calibration' in caplog.text


def test_series_export_uses_canonical_names_and_nulls(tmp_path) -> None:
    records = build_records(
        {
            'date': ['2020-01-01', '2020-01-13'],
            'observed_mean_los_velocity': np.array([np.nan, -0.5]),
            'l2_alarm': np.array([False, True]),
            'l3_level': np.array(['not_evaluated', 'watch']),
        },
        range(2),
    )
    json_path, csv_path = write_series(
        tmp_path / 'monitoring_series', records, rate_unit='mm/day', metadata={}
    )

    payload = json.loads(json_path.read_text())
    assert payload['contract_version'] == '1.0'
    assert payload['series'][0]['observed_mean_los_velocity'] is None
    assert payload['units']['observed_mean_los_velocity'] == 'mm/day'
    assert csv_path.read_text().splitlines()[0] == (
        'date,observed_mean_los_velocity,l2_alarm,l3_level'
    )
    with pytest.raises(KeyError, match='prediction_uncertainty'):
        build_records({'prediction_uncertainty': [1.0]}, range(1))


def test_dag_preprocessing_uses_reference_and_keeps_targets() -> None:
    values = np.array([1.0, 1.0, 100.0, np.nan])[:, np.newaxis, np.newaxis]
    features = {'velocity_lag1': values.copy(), 'velocity_current': values.copy()}
    reference = np.array([True, True, False, False])[:, np.newaxis, np.newaxis]

    filled = handle_missing_values(
        features, {'enabled': True, 'strategy': 'mean'}, reference=reference,
        exclude=frozenset({'velocity_current'}),
    )
    clipped = transform_outliers(
        features, {'enabled': True, 'method': 'clip', 'clip_range': [0.0, 1.0]},
        reference=reference, exclude=frozenset({'velocity_current'}),
    )
    normalized = normalize_features(
        {'velocity_lag1': np.array([1.0, 3.0, 100.0])},
        {'enabled': True, 'method': 'minmax'},
        reference=np.array([True, True, False]),
    )

    # Fill value and clip bounds come from the first two (calibration) dates.
    assert filled['velocity_lag1'][3, 0, 0] == pytest.approx(1.0)
    assert clipped['velocity_lag1'][2, 0, 0] == pytest.approx(1.0)
    assert normalized['velocity_lag1'][1] == pytest.approx(1.0)
    # The forecast target stays a physical observation.
    assert np.isnan(filled['velocity_current'][3, 0, 0])
    assert clipped['velocity_current'][2, 0, 0] == pytest.approx(100.0)
