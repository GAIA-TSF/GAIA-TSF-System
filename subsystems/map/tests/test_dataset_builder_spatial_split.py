"""Tests for pixel-grouped spatial learning splits."""

from __future__ import annotations

import numpy as np
import pytest

from subsystems.map.dataset.dataset_builder import Dataset, DatasetBuilder
from subsystems.map.dataset.feature_loader import RasterGrid


def _dataset(time_count: int = 5, pixel_count: int = 12) -> Dataset:
    time_indices = np.repeat(np.arange(time_count), pixel_count)
    pixel_indices = np.tile(np.arange(pixel_count), time_count)
    return Dataset(
        features=np.column_stack((time_indices, pixel_indices)).astype(float),
        targets=(time_indices + pixel_indices).astype(float),
        time_indices=time_indices,
        pixel_indices=pixel_indices,
        feature_names=('time', 'pixel'),
        dates=tuple(f'2020-01-{index + 1:02d}' for index in range(time_count)),
        grid=RasterGrid(None, None, pixel_count, 1, None),
        mask=np.ones((1, pixel_count), dtype=bool),
    )


def test_spatial_split_keeps_pixel_histories_disjoint_and_all_dates() -> None:
    dataset = _dataset()

    splits = DatasetBuilder().split_spatial_window(
        dataset, 1, 5, 0.5, 0.25, 0.25, random_seed=42
    )

    train_pixels = np.unique(splits.train.pixel_indices)
    validation_pixels = np.unique(splits.validation.pixel_indices)
    test_pixels = np.unique(splits.test.pixel_indices)
    assert (train_pixels.size, validation_pixels.size, test_pixels.size) == (6, 3, 3)
    assert np.intersect1d(train_pixels, validation_pixels).size == 0
    assert np.intersect1d(train_pixels, test_pixels).size == 0
    assert np.intersect1d(validation_pixels, test_pixels).size == 0
    for subset in (splits.train, splits.validation, splits.test):
        np.testing.assert_array_equal(np.unique(subset.time_indices), [1, 2, 3, 4])


def test_spatial_split_is_reproducible() -> None:
    dataset = _dataset()
    builder = DatasetBuilder()

    first = builder.split_spatial_window(
        dataset, 0, 5, 0.5, 0.25, 0.25, random_seed=7
    )
    second = builder.split_spatial_window(
        dataset, 0, 5, 0.5, 0.25, 0.25, random_seed=7
    )

    np.testing.assert_array_equal(first.train.pixel_indices, second.train.pixel_indices)
    np.testing.assert_array_equal(
        first.validation.pixel_indices, second.validation.pixel_indices
    )
    np.testing.assert_array_equal(first.test.pixel_indices, second.test.pixel_indices)


def test_spatial_split_rejects_missing_date_coverage() -> None:
    dataset = _dataset(time_count=3, pixel_count=3)
    include = ~(
        (dataset.time_indices == 2)
        & (dataset.pixel_indices != 0)
    )
    incomplete = DatasetBuilder._subset(dataset, include)

    with pytest.raises(ValueError, match='has no valid pixels'):
        DatasetBuilder().split_spatial_window(
            incomplete, 0, 3, 1 / 3, 1 / 3, 1 / 3, random_seed=1
        )
