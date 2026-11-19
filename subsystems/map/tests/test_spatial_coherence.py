"""Tests for coherence-qualified spatial anomaly products."""

from __future__ import annotations

import numpy as np

from subsystems.map.monitoring.spatial_coherence import SpatialCoherenceDetector


def test_coherence_requires_area_and_temporal_overlap() -> None:
    """Isolated and short-lived anomaly pixels must not become coherent alerts."""
    detector = SpatialCoherenceDetector(
        {
            'enabled': True,
            'minimum_area_pixels': 4,
            'persistence': 2,
            'minimum_overlap': 0.5,
            'connectivity': 4,
        }
    )
    values = np.zeros((3, 5, 5), dtype=bool)
    values[0, 1:3, 1:3] = True
    values[1, 1:3, 1:3] = True
    values[2, 0, 0] = True

    result = detector.detect(values, np.ones((5, 5), dtype=bool))

    assert not np.any(result.binary_stack[0])
    assert np.array_equal(result.binary_stack[1], values[1])
    assert not np.any(result.binary_stack[2])
    assert len(result.regions) == 1
    assert result.regions[0].activation_index == 1
    assert np.array_equal(result.regions[0].support, values[1])
