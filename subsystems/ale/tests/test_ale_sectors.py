"""Sector aggregation with SpatialCoherenceDetector, min_pixels and min_fraction."""

from __future__ import annotations

import numpy as np
import pytest
from ale_3levels.spatial import aggregate_sectors, whole_grid_sectors

COHERENCE = {
    'minimum_area_pixels': 4,
    'persistence': 2,
    'minimum_overlap': 0.3,
    'connectivity': 8,
}


def _stack(times=4, size=10):
    return np.zeros((times, size, size), dtype=np.int8)


def test_persistent_block_raises_the_sector_level():
    severity = _stack()
    severity[1:, 2:5, 2:5] = 3  # 9 pixels at RED/L3 from t=1
    summary = aggregate_sectors(
        severity, whole_grid_sectors(np.ones((10, 10), bool)), COHERENCE, 4, 0.5
    )
    # Coherence needs 2 consecutive acquisitions: level appears at t=2.
    assert summary.severity[:, 0].tolist() == [0, 0, 3, 3]
    assert summary.coherent_counts[2, 0].tolist() == [9, 9, 9]
    assert summary.pixel_counts[2, 0].tolist() == [0, 91, 0, 0, 9]


def test_scattered_pixels_do_not_count():
    severity = _stack()
    severity[:, ::3, ::3] = 3  # isolated pixels, never 4 connected
    summary = aggregate_sectors(
        severity, whole_grid_sectors(np.ones((10, 10), bool)), COHERENCE, 4, 0.5
    )
    assert summary.severity[:, 0].tolist() == [0, 0, 0, 0]


def test_min_fraction_applies_to_small_sectors():
    severity = _stack()
    severity[:, 0:2, 0:3] = 2  # 6 coherent pixels
    small = np.zeros((10, 10), bool)
    small[0:2, 0:5] = True  # sector of 10 valid pixels, 6 at >= 2
    sectors = {'small': small, 'big': ~small}
    summary = aggregate_sectors(severity, sectors, COHERENCE, 50, 0.5)
    assert summary.severity[1:, 0].tolist() == [2, 2, 2]  # 6/10 >= 0.5
    assert summary.severity[1:, 1].tolist() == [0, 0, 0]


def test_sector_without_valid_pixels_is_no_data():
    severity = _stack()
    severity[:, :, :5] = -1
    sectors = {'left': np.zeros((10, 10), bool), 'right': np.zeros((10, 10), bool)}
    sectors['left'][:, :5] = True
    sectors['right'][:, 5:] = True
    summary = aggregate_sectors(severity, sectors, COHERENCE, 4, 0.5)
    assert np.all(summary.severity[:, 0] == -1)
    assert np.all(summary.severity[:, 1] == 0)


def test_invalid_rule_is_rejected():
    with pytest.raises(ValueError):
        aggregate_sectors(
            _stack(), whole_grid_sectors(np.ones((10, 10), bool)), COHERENCE, 0, 0.5
        )
