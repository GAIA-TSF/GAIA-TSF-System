"""Tests for causal temporal alignment of external MAP feature stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from subsystems.map.dataset.feature_loader import FeatureLoader


def test_exact_alignment_selects_daily_meteo_at_acquisition_dates() -> None:
    """Daily meteorology is selected at each exact InSAR acquisition date."""
    loader = FeatureLoader((Path('/tmp/features'),), Path('/tmp/mask.tif'), 'exact')
    values = np.arange(5, dtype=float).reshape(5, 1, 1)

    aligned = loader._align_to_reference_dates(
        values,
        (
            '2020-01-01',
            '2020-01-02',
            '2020-01-03',
            '2020-01-04',
            '2020-01-05',
        ),
        ('2020-01-01', '2020-01-03', '2020-01-05'),
        'precipitation',
    )

    assert np.array_equal(aligned[:, 0, 0], np.array([0.0, 2.0, 4.0]))


def test_previous_alignment_never_uses_future_meteorological_data() -> None:
    """Sparse external features use only data available on or before the date."""
    loader = FeatureLoader((Path('/tmp/features'),), Path('/tmp/mask.tif'), 'previous')
    values = np.array([[[10.0]], [[20.0]]])

    aligned = loader._align_to_reference_dates(
        values,
        ('2020-01-01', '2020-01-04'),
        ('2020-01-02', '2020-01-05'),
        'temperature_mean',
    )

    assert np.array_equal(aligned[:, 0, 0], np.array([10.0, 20.0]))
