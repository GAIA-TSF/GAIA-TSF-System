"""Tests for the L2 residual CUSUM contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from subsystems.map.monitoring.level2 import (
    Level2Parameters,
    aggregate_residual,
    calibration_replay,
    compute_level2,
    frozen_level2_calibration,
    level2_alarms,
    level2_metadata,
    pixel_level2,
    residual_cusum,
)
from subsystems.map.monitoring.spatial_coherence import SpatialCoherenceRegion

PARAMETERS = Level2Parameters(
    reference_value=0.5, decision_threshold=5.0, direction='negative'
)


def _dates(count: int) -> tuple[str, ...]:
    start = np.datetime64('2020-01-01')
    return tuple(str(start + np.timedelta64(12 * index, 'D')) for index in range(count))


def test_cusum_accumulates_standardized_residuals_in_instability_direction() -> None:
    z = np.array([0.0, -1.0, -1.0, -1.0, 1.0])

    instability, overforecast = residual_cusum(z, 0.5, 'negative', 1)

    assert np.isnan(instability[0]) and np.isnan(overforecast[0])
    assert np.allclose(instability[1:], [0.5, 1.0, 1.5, 0.0])
    assert np.allclose(overforecast[1:], [0.0, 0.0, 0.0, 0.5])


def test_missing_residual_holds_the_statistic_and_there_is_no_reset() -> None:
    z = np.array([-3.0, np.nan, -3.0, -3.0, -3.0])

    instability, _ = residual_cusum(z, 0.5, 'negative', 0)
    alarm, first = level2_alarms(instability, 5.0)

    assert instability[1] == instability[0] == 2.5
    assert first == 3
    assert alarm.tolist() == [False, False, False, True, True]
    assert instability[4] > instability[3]


def test_alarm_comparison_is_strict() -> None:
    alarm, first = level2_alarms(np.array([4.9, 5.0, 5.1]), 5.0)

    assert alarm.tolist() == [False, False, True]
    assert first == 2


def test_single_spike_below_threshold_does_not_alarm() -> None:
    z = np.zeros(30)
    z[10] = -4.0

    instability, _ = residual_cusum(z, 0.5, 'negative', 0)

    assert level2_alarms(instability, 5.0)[1] is None


def test_one_sigma_step_is_detected_within_expected_delay() -> None:
    rng = np.random.default_rng(3)
    delays = []
    for _ in range(200):
        z = rng.standard_normal(200)
        z[100:] -= 1.0
        instability, _ = residual_cusum(z, 0.5, 'negative', 100)
        first = level2_alarms(instability, 5.0)[1]
        assert first is not None
        delays.append(first - 100)
    # Siegmund approximation for k=0.5, h=5, shift 1: about 10.4 samples.
    assert 7.0 < np.mean(delays) < 14.0


def test_in_control_false_alarm_rate_is_low() -> None:
    rng = np.random.default_rng(11)
    alarms = 0
    for _ in range(200):
        instability, _ = residual_cusum(rng.standard_normal(100), 0.5, 'negative', 0)
        alarms += level2_alarms(instability, 5.0)[1] is not None
    # ARL0 is about 465 for iid z, so roughly 19% of 100-sample runs alarm.
    assert alarms / 200 < 0.35


def test_positive_direction_mirrors_negative() -> None:
    z = np.array([1.0, 2.0, -1.0])

    negative = residual_cusum(-z, 0.5, 'negative', 0)
    positive = residual_cusum(z, 0.5, 'positive', 0)

    assert np.allclose(negative[0], positive[0])
    assert np.allclose(negative[1], positive[1])


def test_parameters_reject_two_sided_direction_and_read_l1_default() -> None:
    with pytest.raises(ValueError, match='level2.direction'):
        Level2Parameters.from_config({'anomaly_detection': {'direction': 'both'}})
    parameters = Level2Parameters.from_config(
        {'anomaly_detection': {'direction': 'negative'}, 'level2': {}}
    )
    assert parameters == PARAMETERS


def test_aggregate_uses_fixed_support_not_per_date_tail() -> None:
    stack = np.array([[[1.0, -5.0], [np.nan, 3.0]]])
    support = np.array([[True, True], [True, False]])

    assert np.allclose(aggregate_residual(stack, support), [-2.0])


def _calibrated(tmp_path, stack, dates, mask, calibration=(0, 10), start=10):
    return frozen_level2_calibration(
        tmp_path / 'l2.json', stack, dates, calibration, start, mask,
        unit='mm/day', target_feature='velocity', parameters=PARAMETERS,
    )


def test_frozen_calibration_uses_pre_monitoring_dates_and_rejects_changes(
    tmp_path,
) -> None:
    rng = np.random.default_rng(5)
    dates = _dates(20)
    mask = np.ones((3, 3), dtype=bool)
    stack = rng.normal(0.0, 1.0, (20, 3, 3))

    # Calibration window overlaps monitoring; only dates before index 10 fit.
    payload = _calibrated(tmp_path, stack, dates, mask, calibration=(0, 15))
    expected = np.std(aggregate_residual(stack, mask)[:10], ddof=1)

    assert payload['sigma'] == pytest.approx(expected)
    assert payload['last_date'] == dates[9]
    assert payload['method'] == 'aggregate_residual_sample_std_ddof1_no_demeaning'
    # Monitoring data changes do not refit sigma.
    changed_monitoring = stack.copy()
    changed_monitoring[12] += 100.0
    assert _calibrated(tmp_path, changed_monitoring, dates, mask) == payload
    changed_history = stack.copy()
    changed_history[2] += 1.0
    with pytest.raises(ValueError, match='history/support changed'):
        _calibrated(tmp_path, changed_history, dates, mask)


def test_calibration_records_bias_and_warns(tmp_path, caplog) -> None:
    rng = np.random.default_rng(9)
    dates = _dates(20)
    mask = np.ones((1, 1), dtype=bool)
    stack = (rng.normal(0.0, 1.0, 20) - 1.0)[:, np.newaxis, np.newaxis]

    payload = _calibrated(tmp_path, stack, dates, mask)

    assert payload['bias_ratio'] == pytest.approx(
        payload['mean_residual'] / payload['sigma']
    )
    assert abs(payload['bias_ratio']) > 0.25
    assert 'L2 calibration bias' in caplog.text
    assert json.loads((tmp_path / 'l2.json').read_text())['sigma'] == payload['sigma']


def test_compute_level2_is_causal_and_reports_first_alarm(tmp_path) -> None:
    rng = np.random.default_rng(2)
    dates = _dates(40)
    mask = np.ones((2, 2), dtype=bool)
    stack = rng.normal(0.0, 0.1, (40, 2, 2))
    stack[25:] -= 0.3
    calibration = _calibrated(tmp_path, stack, dates, mask)

    full = compute_level2(stack, mask, (0, 10), (10, 40), calibration, PARAMETERS)
    prefix = compute_level2(
        stack[:30], mask, (0, 10), (10, 30), calibration, PARAMETERS
    )

    assert np.all(np.isnan(full.instability_cusum[:10]))
    assert np.allclose(
        full.instability_cusum[:30], prefix.instability_cusum, equal_nan=True
    )
    assert full.first_alarm_index is not None and full.first_alarm_index >= 25
    metadata = level2_metadata(full, dates)
    assert metadata['first_alarm_date'] == dates[full.first_alarm_index]
    assert metadata['reset_after_alarm'] is False
    assert metadata['calibration_replay']['samples'] == 10


def test_regional_cusum_uses_own_sigma_and_starts_at_activation(tmp_path) -> None:
    rng = np.random.default_rng(4)
    dates = _dates(30)
    mask = np.ones((4, 4), dtype=bool)
    stack = rng.normal(0.0, 0.1, (30, 4, 4))
    support = np.zeros((4, 4), dtype=bool)
    support[0, 0] = True
    stack[12:, 0, 0] -= 1.0
    calibration = _calibrated(tmp_path, stack, dates, mask)
    region = SpatialCoherenceRegion(support=support, activation_index=15)

    result = compute_level2(
        stack, mask, (0, 10), (10, 30), calibration, PARAMETERS, (region,)
    )

    assert not np.any(result.regional_available[:15])
    assert np.all(result.regional_available[15:])
    assert result.regional_instability_cusum[15] < 15.0
    assert result.regional_first_alarm_index is not None
    region_sigma = result.regional_sigmas[0]['sigma']
    assert region_sigma == pytest.approx(np.std(stack[:10, 0, 0], ddof=1))


def test_pixel_level2_accumulates_l1_z() -> None:
    stack = np.zeros((6, 1, 2))
    stack[2:, 0, 0] = -2.0
    mask = np.array([[True, True]])

    maximum, first_alarm = pixel_level2(stack, 1.0, mask, (2, 6), PARAMETERS)

    assert maximum[0, 0] == pytest.approx(6.0)
    assert first_alarm[0, 0] == 5
    assert maximum[0, 1] == 0.0 and np.isnan(first_alarm[0, 1])


def test_calibration_replay_counts_episodes() -> None:
    z = np.r_[np.full(8, -2.0), np.full(20, 3.0), np.full(8, -2.0)]

    assert calibration_replay(z, PARAMETERS) == {'samples': 36, 'alarm_episodes': 2}
