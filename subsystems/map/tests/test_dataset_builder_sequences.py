"""Tests for causal sequence construction used by sequence-model plugins."""

from __future__ import annotations

import numpy as np

from subsystems.map.dataset.dataset_builder import Dataset, DatasetBuilder
from subsystems.map.dataset.feature_loader import RasterGrid


def test_build_sequences_uses_only_prior_acquisitions() -> None:
    """A one-step forecast excludes the target acquisition from inputs."""
    time_indices = np.repeat(np.arange(4), 2)
    pixel_indices = np.tile(np.array([0, 1]), 4)
    features = np.column_stack((time_indices.astype(float), pixel_indices.astype(float)))
    dataset = Dataset(
        features=features,
        targets=time_indices.astype(float),
        time_indices=time_indices,
        pixel_indices=pixel_indices,
        feature_names=('time_value', 'pixel_value'),
        dates=('2020-01-01', '2020-01-13', '2020-01-25', '2020-02-06'),
        grid=RasterGrid(None, None, 1, 2, None),
        mask=np.array([[True, True]]),
    )

    sequences = DatasetBuilder().build_sequences(dataset, look_back=2, horizon=1)

    assert sequences.features.shape == (4, 2, 2)
    assert np.array_equal(sequences.time_indices, np.array([2, 2, 3, 3]))
    assert np.array_equal(sequences.targets, np.array([2.0, 2.0, 3.0, 3.0]))
    assert np.array_equal(sequences.features[0, :, 0], np.array([0.0, 1.0]))
    assert np.array_equal(sequences.features[-1, :, 0], np.array([1.0, 2.0]))
