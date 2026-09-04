"""Tests for TSF acceleration monitoring signals."""

from __future__ import annotations

import numpy as np

from subsystems.map.monitoring.temporal_monitoring import TemporalResidualMonitor


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
