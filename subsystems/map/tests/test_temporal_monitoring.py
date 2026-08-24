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
