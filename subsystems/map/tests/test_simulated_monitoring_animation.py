"""Focused tests for the causal monitoring-animation orchestration."""

from __future__ import annotations

import numpy as np

from subsystems.map.scripts.simulate_monitoring_animation import (
    causal_prefix_indices,
    classify_risk,
    frame_indices,
)
from subsystems.map.utils.temporal_windows import resolve_temporal_window


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
