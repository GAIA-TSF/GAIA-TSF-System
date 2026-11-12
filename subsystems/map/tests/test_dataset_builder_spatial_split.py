"""Pixel-grouped spatial splits and causal forecast-sample construction."""

from __future__ import annotations

import numpy as np
import pytest

from subsystems.map.dataset import LoadedFeatures, RasterGrid, build_forecast_samples, split_spatial


def _loaded(time_count: int = 6, pixel_count: int = 12) -> LoadedFeatures:
    time = np.arange(time_count, dtype=float)[:, None, None]
    pixel = np.arange(pixel_count, dtype=float)[None, None, :]
    target = np.broadcast_to(time * 10 + pixel, (time_count, 1, pixel_count)).copy()
    return LoadedFeatures(
        {'target': target, 'feature': target.copy()},
        tuple(f'2020-01-{index + 1:02d}' for index in range(time_count)),
        RasterGrid(None, (1, 0, 0, 0, -1, 0), 1, pixel_count, None),
        np.ones((1, pixel_count), dtype=bool),
    )


def test_samples_use_only_origin_and_earlier_acquisitions() -> None:
    samples = build_forecast_samples(_loaded(), ['feature'], 'target', 2, look_back=3)
    # The sequence ends at the origin; targets are the next two acquisitions.
    origin, pixel = samples.origin_indices, samples.pixel_indices
    np.testing.assert_array_equal(samples.features[:, -1, 0], origin * 10 + pixel)
    np.testing.assert_array_equal(samples.features[:, 0, 0], (origin - 2) * 10 + pixel)
    np.testing.assert_array_equal(samples.targets[:, 0], np.where(origin + 1 < 6, (origin + 1) * 10 + pixel, np.nan))
    assert origin.min() == 2


def test_spatial_split_keeps_pixel_histories_disjoint_and_all_dates() -> None:
    samples = build_forecast_samples(_loaded(), ['feature'], 'target', 1)
    splits = split_spatial(samples, 0, 5, 0.5, 0.25, 0.25, random_seed=7)
    pixels = [set(subset.pixel_indices) for _, subset in splits.items()]
    assert not (pixels[0] & pixels[1] or pixels[0] & pixels[2] or pixels[1] & pixels[2])
    for _, subset in splits.items():
        np.testing.assert_array_equal(np.unique(subset.origin_indices), np.arange(5))
        # Targets at or beyond the window end are purged.
        assert np.all(np.isnan(subset.targets[subset.origin_indices == 4]))


def test_spatial_split_is_reproducible() -> None:
    samples = build_forecast_samples(_loaded(), ['feature'], 'target', 1)
    first = split_spatial(samples, 0, 5, 0.5, 0.25, 0.25, random_seed=3)
    second = split_spatial(samples, 0, 5, 0.5, 0.25, 0.25, random_seed=3)
    np.testing.assert_array_equal(first.train.pixel_indices, second.train.pixel_indices)


def test_spatial_split_rejects_missing_date_coverage() -> None:
    loaded = _loaded(pixel_count=4)
    loaded.features['feature'][2, 0, 1:] = np.nan
    samples = build_forecast_samples(loaded, ['feature'], 'target', 1)
    with pytest.raises(ValueError, match='has no valid pixels'):
        split_spatial(samples, 0, 5, 0.25, 0.25, 0.5, random_seed=0)
