"""Unit tests for MAP explainability helpers."""

from __future__ import annotations

import numpy as np

from subsystems.map.utils.explainability import (
    grouped_permutation_importance,
    temporal_stratified_sample_indices,
)


class _FirstFeatureEstimator:
    """Deterministic estimator used to test permutation scoring."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the first explanatory column."""
        return features[:, 0]


def test_temporal_stratified_sample_indices_represents_each_time() -> None:
    """Sampling uses every acquisition when capacity permits it."""
    time_indices = np.repeat(np.arange(4), 10)

    sample = temporal_stratified_sample_indices(time_indices, 8, random_seed=42)

    assert sample.size == 8
    assert set(time_indices[sample]) == {0, 1, 2, 3}
    assert np.array_equal(
        sample,
        temporal_stratified_sample_indices(time_indices, 8, random_seed=42),
    )


def test_grouped_permutation_importance_preserves_group_membership() -> None:
    """A predictive group has a positive held-out RMSE increase."""
    features = np.column_stack((np.arange(20, dtype=float), np.ones(20)))
    importance, skipped = grouped_permutation_importance(
        _FirstFeatureEstimator(),
        features,
        features[:, 0],
        ['signal', 'noise'],
        {'signal_group': ['signal'], 'noise_group': ['noise']},
        n_repeats=4,
        random_seed=42,
        value_scale=1000.0,
    )

    by_group = {item['group']: item for item in importance}
    assert by_group['signal_group']['rmse_increase_mean'] > 0.0
    assert by_group['noise_group']['rmse_increase_mean'] == 0.0
    assert skipped == {}


def test_grouped_permutation_importance_is_reproducible_in_parallel() -> None:
    """Parallel group evaluation yields the same deterministic values as serial."""
    features = np.column_stack((np.arange(20, dtype=float), np.ones(20)))
    arguments = (
        _FirstFeatureEstimator(),
        features,
        features[:, 0],
        ['signal', 'noise'],
        {'signal_group': ['signal'], 'noise_group': ['noise']},
        4,
        42,
        1000.0,
    )

    serial, _ = grouped_permutation_importance(*arguments, n_jobs=1)
    parallel, _ = grouped_permutation_importance(*arguments, n_jobs=2)

    assert serial == parallel
