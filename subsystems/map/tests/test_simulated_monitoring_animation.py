"""Focused tests for the causal monitoring-animation orchestration."""

from __future__ import annotations

import numpy as np

from subsystems.map.scripts.simulate_monitoring_animation import (
    causal_prefix_indices,
    classify_risk,
    configured_animation_output,
    frame_indices,
    run_precomputed_simulation,
)
from subsystems.map.utils.temporal_windows import TemporalWindow, resolve_temporal_window


DATES = (
    '2019-12-01',
    '2019-12-15',
    '2020-01-01',
    '2020-01-13',
    '2020-01-25',
    '2020-02-06',
)
WINDOWS = {
    'temporal_windows': {
        'calibration': {
            'start_date': '2019-12-01',
            'end_date': '2020-01-01',
        },
        'monitoring': {
            'start_date': '2020-01-01',
            'end_date': '2020-01-25',
        },
    }
}


def test_calibration_is_end_exclusive_and_monitoring_stops_at_end() -> None:
    calibration = resolve_temporal_window(
        DATES, WINDOWS, 'calibration', end_inclusive=False
    )
    monitoring = resolve_temporal_window(DATES, WINDOWS, 'monitoring')

    assert DATES[calibration.start_index : calibration.end_index] == (
        '2019-12-01',
        '2019-12-15',
    )
    assert DATES[monitoring.start_index : monitoring.end_index] == (
        '2020-01-01',
        '2020-01-13',
        '2020-01-25',
    )


def test_causal_prefix_excludes_future_samples() -> None:
    sample_times = np.array([0, 0, 1, 2, 2, 3, 4])

    selected = causal_prefix_indices(sample_times, current_index=2)

    assert np.array_equal(selected, np.array([0, 1, 2, 3, 4]))
    assert np.all(sample_times[selected] <= 2)


def test_risk_threshold_classification_uses_inclusive_boundaries() -> None:
    assert classify_risk(0.29, 0.3, 0.7) == 'NORMAL'
    assert classify_risk(0.3, 0.3, 0.7) == 'MEDIUM'
    assert classify_risk(0.69, 0.3, 0.7) == 'MEDIUM'
    assert classify_risk(0.7, 0.3, 0.7) == 'HIGH'


def test_frame_count_uses_only_real_acquisition_dates() -> None:
    calibration = resolve_temporal_window(
        DATES, WINDOWS, 'calibration', end_inclusive=False
    )
    monitoring = resolve_temporal_window(DATES, WINDOWS, 'monitoring')

    indices = frame_indices(DATES, calibration, monitoring)

    assert indices == (0, 1, 2, 3, 4)
    assert len(indices) == 2 + 3


def test_animation_output_is_resolved_below_experiment_results(tmp_path) -> None:
    """The configured animation directory is scenario-local by default."""
    config = {
        '_config_path': tmp_path / 'config.yaml',
        'experiment_dir': str(tmp_path / 'synthetic_scenario'),
        'monitoring': {
            'animation': {
                'output_directory': 'monitoring/animation',
                'filename': 'map_monitoring.mp4',
            },
        },
    }

    output = configured_animation_output(config)

    assert output == (
        tmp_path
        / 'synthetic_scenario'
        / 'results'
        / 'monitoring'
        / 'animation'
        / 'map_monitoring.mp4'
    )


def test_precomputed_replay_supports_overlapping_spatial_calibration() -> None:
    """A full-period spatial calibration produces one frame per monitoring date."""
    dates = tuple(f'2020-01-{day:02d}' for day in range(1, 9))
    observed = np.arange(8, dtype=float)[:, np.newaxis, np.newaxis]
    config = {
        'monitoring': {
            'dashboard': {
                'anomaly_magnitude_threshold': 0.1,
                'cusum': {
                    'instability_direction': 'negative',
                    'signal': 'observed_velocity',
                    'spatial_aggregation': 'mean',
                    'spatial_quantile': 0.1,
                    'reference_value': 0.5,
                    'decision_threshold': 2.0,
                    'derivative_window': 2,
                    'smoothing_span': 2,
                    'persistence_window': 2,
                    'persistence_threshold': 0.25,
                },
                'regime': {
                    'signal': 'unexpected_acceleration',
                    'smoothing_span': 2,
                    'medium_risk_threshold': 0.3,
                    'high_risk_threshold': 0.7,
                },
            }
        },
        'plotting': {},
    }
    shared_window = TemporalWindow(0, 8, dates[0], dates[-1])

    result = run_precomputed_simulation(
        config,
        dates=dates,
        calibration=shared_window,
        monitoring=shared_window,
        observed_stack=observed,
        prediction_stack=np.zeros_like(observed),
        uncertainty_stack=None,
        fixed_support_mask=None,
    )

    assert len(result.frames) == 8
    assert result.frames[0].date == dates[0]
    assert result.frames[-1].date == dates[-1]
