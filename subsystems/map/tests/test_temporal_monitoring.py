"""Tests for TSF acceleration monitoring signals."""

from __future__ import annotations

import numpy as np

from subsystems.map.monitoring.temporal_monitoring import TemporalResidualMonitor
from subsystems.map.monitoring.dashboard import _cusum_status
from subsystems.map.dataset.dataset_builder import DatasetBuilder


def test_observed_velocity_cusum_detects_negative_acceleration() -> None:
    """Physical acceleration remains visible when model residuals are zero."""
    monitor = TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'cusum': {
                'instability_direction': 'negative',
                'signal': 'observed_velocity',
                'reference_value': 0.5,
                'decision_threshold': 2.0,
                'smoothing_span': 2,
                'persistence_window': 2,
                'persistence_threshold': 0.25,
            },
            'regime': {
                'smoothing_span': 2,
                'medium_risk_threshold': 0.3,
                'high_risk_threshold': 0.7,
            },
        },
    )
    velocity = np.array([0.0, -1.0, -2.0, -3.0, -5.0, -8.0, -12.0, -17.0])
    stack = velocity[:, np.newaxis, np.newaxis]

    result = monitor.analyze(
        observed_stack=stack,
        prediction_stack=stack.copy(),
        dates=tuple(f'2020-01-{day:02d}' for day in range(1, 9)),
        calibration_window=(0, 4),
        monitoring_window=(4, 8),
    )

    assert np.allclose(result.residual_mean, 0.0)
    assert result.acceleration_cusum[-1] > result.deceleration_cusum[-1]
    assert result.acceleration_cusum[-1] > 2.0


def test_regime_risk_ignores_seasonality_predicted_by_baseline() -> None:
    """A recurring predicted seasonal cycle must not create regime evidence."""
    monitor = TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'cusum': {
                'instability_direction': 'negative',
                'signal': 'observed_velocity',
                'reference_value': 0.5,
                'decision_threshold': 2.0,
                'smoothing_span': 3,
                'persistence_window': 3,
                'persistence_threshold': 0.25,
            },
            'regime': {
                'smoothing_span': 3,
                'medium_risk_threshold': 0.3,
                'high_risk_threshold': 0.7,
            },
        },
    )
    positions = np.linspace(0.0, 4.0 * np.pi, 24)
    seasonal_velocity = np.sin(positions)
    stack = seasonal_velocity[:, np.newaxis, np.newaxis]

    result = monitor.analyze(
        observed_stack=stack,
        prediction_stack=stack.copy(),
        dates=tuple(f'2020-01-{day:02d}' for day in range(1, 25)),
        calibration_window=(0, 12),
        monitoring_window=(12, 24),
    )

    assert np.allclose(result.regime_risk[12:], 0.0)


def test_regime_risk_detects_acceleration_not_predicted_by_baseline() -> None:
    """Unexpected negative acceleration produces regime-change evidence."""
    monitor = TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'cusum': {
                'instability_direction': 'negative',
                'signal': 'observed_velocity',
                'reference_value': 0.5,
                'decision_threshold': 2.0,
                'smoothing_span': 2,
                'persistence_window': 2,
                'persistence_threshold': 0.25,
            },
            'regime': {
                'smoothing_span': 2,
                'medium_risk_threshold': 0.3,
                'high_risk_threshold': 0.7,
            },
        },
    )
    predicted = np.zeros(12)
    observed = predicted.copy()
    observed[6:] = -np.square(np.arange(1, 7, dtype=float))

    result = monitor.analyze(
        observed_stack=observed[:, np.newaxis, np.newaxis],
        prediction_stack=predicted[:, np.newaxis, np.newaxis],
        dates=tuple(f'2020-01-{day:02d}' for day in range(1, 13)),
        calibration_window=(0, 6),
        monitoring_window=(6, 12),
    )

    assert result.regime_risk[-1] > 0.7


def test_monitoring_signals_do_not_change_when_future_data_arrives() -> None:
    """Causal CUSUM and regime values match a replay frame at the same date."""
    monitor = TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'cusum': {
                'instability_direction': 'negative',
                'signal': 'observed_velocity',
                'reference_value': 0.5,
                'decision_threshold': 2.0,
                'derivative_window': 3,
                'smoothing_span': 3,
                'persistence_window': 3,
                'persistence_threshold': 0.25,
            },
            'regime': {
                'smoothing_span': 3,
                'medium_risk_threshold': 0.3,
                'high_risk_threshold': 0.7,
            },
        },
    )
    observed = np.array([0.0, 0.0, -0.1, -0.2, -0.5, -1.0, -1.8, -3.0, -4.5, -6.5])
    predicted = np.zeros(observed.size)
    dates = tuple(f'2020-01-{day:02d}' for day in range(1, observed.size + 1))

    full = monitor.analyze(
        observed[:, np.newaxis, np.newaxis],
        predicted[:, np.newaxis, np.newaxis],
        dates,
        calibration_window=(0, 4),
        monitoring_window=(4, observed.size),
    )
    current_index = 8
    replay = monitor.analyze(
        observed[: current_index + 1, np.newaxis, np.newaxis],
        predicted[: current_index + 1, np.newaxis, np.newaxis],
        dates[: current_index + 1],
        calibration_window=(0, 4),
        monitoring_window=(4, current_index + 1),
    )

    assert replay.acceleration_cusum[current_index] == full.acceleration_cusum[current_index]
    assert replay.deceleration_cusum[current_index] == full.deceleration_cusum[current_index]
    assert replay.regime_risk[current_index] == full.regime_risk[current_index]


def test_directional_tail_cusum_detects_local_negative_acceleration() -> None:
    """A local negative failure signal is not diluted by stable TSF pixels."""
    monitor = TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'cusum': {
                'instability_direction': 'negative',
                'signal': 'observed_velocity',
                'spatial_aggregation': 'directional_tail_mean',
                'spatial_quantile': 0.10,
                'reference_value': 0.5,
                'decision_threshold': 2.0,
                'derivative_window': 3,
                'smoothing_span': 2,
                'persistence_window': 2,
                'persistence_threshold': 0.25,
            },
            'regime': {
                'smoothing_span': 2,
                'medium_risk_threshold': 0.3,
                'high_risk_threshold': 0.7,
            },
        },
    )
    stable = np.tile(
        np.array([0.0, 0.1, -0.1, 0.0, 0.2, 0.3, 0.4, 0.5]),
        (9, 1),
    ).T
    local_failure = np.array([0.0, 0.1, -0.1, 0.0, -1.0, -3.0, -6.0, -10.0])
    stack = np.column_stack((stable, local_failure))[:, np.newaxis, :]
    dates = tuple(f'2020-01-{day:02d}' for day in range(1, 9))

    result = monitor.analyze(
        stack,
        stack.copy(),
        dates,
        calibration_window=(0, 4),
        monitoring_window=(4, 8),
    )

    assert result.acceleration_cusum[-1] > result.deceleration_cusum[-1]
    assert result.acceleration_cusum[-1] > 2.0
    assert _cusum_status(result, -1) == 'Acceleration alarm'


def test_fixed_valid_support_is_based_on_observations_not_model_features() -> None:
    """A missing model feature must not alter the physical support mask."""
    target = np.ones((3, 2, 2), dtype=float)
    target[1, 0, 1] = np.nan
    mask = np.ones((2, 2), dtype=bool)

    support = DatasetBuilder.fixed_valid_mask(target, mask, 0, 3)

    assert np.array_equal(support, np.array([[True, False], [True, True]]))


def test_shared_support_aligns_observed_and_predicted_spatial_means() -> None:
    """Observed and predicted aggregates exclude unmatched pixels together."""
    monitor = TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'cusum': {
                'signal': 'observed_velocity',
                'spatial_aggregation': 'mean',
                'instability_direction': 'negative',
                'reference_value': 0.1,
                'decision_threshold': 2.0,
                'smoothing_span': 2,
                'persistence_window': 2,
                'persistence_threshold': 0.25,
            },
            'regime': {'smoothing_span': 2, 'medium_risk_threshold': 0.3, 'high_risk_threshold': 0.7},
        },
    )
    observed = np.array(
        [[[1.0, 9.0]], [[2.0, 10.0]], [[3.0, 11.0]], [[4.0, 12.0]], [[5.0, 13.0]]]
    )
    predicted = np.array(
        [[[1.0, np.nan]], [[2.0, np.nan]], [[3.0, np.nan]], [[4.0, np.nan]], [[5.0, np.nan]]]
    )

    result = monitor.analyze(
        observed,
        predicted,
        dates=tuple(f'2020-01-0{index}' for index in range(1, 6)),
        calibration_window=(0, 4),
        monitoring_window=(4, 5),
    )

    assert np.allclose(result.observed_mean, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(result.predicted_mean, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_observed_velocity_cusum_is_independent_of_prediction_stack() -> None:
    """Physical CUSUM must not change when a model's predictions change."""
    monitor = TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'cusum': {
                'signal': 'observed_velocity',
                'instability_direction': 'negative',
                'reference_value': 0.1,
                'decision_threshold': 2.0,
                'derivative_window': 2,
                'smoothing_span': 2,
                'persistence_window': 2,
                'persistence_threshold': 0.25,
            },
            'regime': {'smoothing_span': 2, 'medium_risk_threshold': 0.3, 'high_risk_threshold': 0.7},
        },
    )
    observed = np.array([0.0, -0.1, -0.4, -0.9, -1.6, -2.5, -3.6])[:, np.newaxis, np.newaxis]
    first = monitor.analyze(
        observed,
        np.zeros_like(observed),
        tuple(f'2020-01-0{index}' for index in range(1, 8)),
        calibration_window=(0, 4),
        monitoring_window=(4, 7),
    )
    second = monitor.analyze(
        observed,
        np.full_like(observed, 100.0),
        tuple(f'2020-01-0{index}' for index in range(1, 8)),
        calibration_window=(0, 4),
        monitoring_window=(4, 7),
    )

    assert np.allclose(first.acceleration_cusum, second.acceleration_cusum)
    assert np.allclose(first.deceleration_cusum, second.deceleration_cusum)
