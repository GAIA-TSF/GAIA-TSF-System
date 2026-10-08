"""Tests for clean-water variability noise detection."""

import numpy as np
import pytest

from subsystems.dag.plugins.eda.amd_noise import clean_water_variability, spatial_inconsistency


def test_clean_water_variability_flags_spatially_noisy_date():
    data = np.array([
        [[1, 1, 1, 1]],
        [[1, 1.1, .9, 1]],
        [[-10, 10, -8, 8]],
        [[1, 1.2, .8, 1]],
    ], dtype=float)
    result = clean_water_variability(
        data, np.ones((1, 4), dtype=bool), threshold_sigma=3,
        minimum_valid_pixels=4,
    )
    np.testing.assert_array_equal(result['noisy'], [False, False, True, False])
    assert result['threshold'] < result['robust_sigma'][2]


def test_clean_water_variability_requires_enough_pixels():
    with pytest.raises(ValueError, match='enough valid'):
        clean_water_variability(
            np.ones((2, 1, 2)), np.ones((1, 2), dtype=bool),
            minimum_valid_pixels=3,
        )


def test_spatial_inconsistency_flags_isolated_pixel():
    data = np.ones((1, 5, 5), dtype=float)
    data[0, 2, 2] = 20
    result = spatial_inconsistency(
        data, np.ones((5, 5), dtype=bool), window_size=3,
        threshold_sigma=3, minimum_valid_neighbors=3,
        minimum_reference_pixels=10,
    )
    assert result['flags'][0, 2, 2]
    assert result['flags'].sum() == 1
