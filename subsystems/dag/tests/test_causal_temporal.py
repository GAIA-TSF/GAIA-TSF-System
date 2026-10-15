"""Tests for past-only Sentinel-1 temporal derivatives."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from subsystems.dag.utils.temporal import causal_temporal_gradient


def test_causal_temporal_gradient_uses_irregular_backward_intervals() -> None:
    values = np.array([0.0, 10.0, 22.0], dtype=np.float32)[:, None, None]
    dates = (date(2020, 1, 1), date(2020, 1, 6), date(2020, 1, 12))
    velocity = causal_temporal_gradient(values, dates, order=1)
    assert np.isnan(velocity[0, 0, 0])
    assert np.allclose(velocity[1:, 0, 0], [2.0, 2.0])


def test_second_causal_derivative_has_two_undefined_initial_layers() -> None:
    values = np.array([0.0, 10.0, 22.0], dtype=np.float32)[:, None, None]
    dates = (date(2020, 1, 1), date(2020, 1, 6), date(2020, 1, 12))
    acceleration = causal_temporal_gradient(values, dates, order=2)
    assert np.all(np.isnan(acceleration[:2]))
    assert np.isclose(acceleration[2, 0, 0], 0.0)


def test_causal_temporal_gradient_is_strictly_causal_under_future_acquisitions() -> None:
    """Historical derivatives must not change when future acquisitions are appended."""
    values_initial = np.array([0.0, 10.0, 22.0], dtype=np.float32)[:, None, None]
    dates_initial = (date(2020, 1, 1), date(2020, 1, 6), date(2020, 1, 12))
    velocity_initial = causal_temporal_gradient(values_initial, dates_initial, order=1)

    # Future acquisition arrives with large sudden jump
    values_extended = np.array([0.0, 10.0, 22.0, 100.0], dtype=np.float32)[:, None, None]
    dates_extended = (date(2020, 1, 1), date(2020, 1, 6), date(2020, 1, 12), date(2020, 1, 18))
    velocity_extended = causal_temporal_gradient(values_extended, dates_extended, order=1)

    # The past values must be identical
    assert np.isnan(velocity_extended[0, 0, 0])
    assert np.allclose(velocity_extended[:3, 0, 0], velocity_initial[:, 0, 0], equal_nan=True)

