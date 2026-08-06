from __future__ import annotations

from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from subsystems.dag.plugins.ingestion.sentinel1_loader import Sentinel1LOSLoader
from subsystems.dag.plugins.features.slope_features import SlopeFeatureExtractor
from subsystems.dag.plugins.features.temporal_features import TemporalFeatureExtractor
from subsystems.dag.utils.raster import apply_mask
from subsystems.dag.utils.statistics import time_series_statistics


def _write_raster(path, values):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype='float32',
        crs='EPSG:32633',
        transform=from_origin(0, 10, 10, 10),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def test_sentinel1_loader_sorts_by_acquisition_date(tmp_path):
    _write_raster(tmp_path / 'tsf_los_20180113.tif', np.ones((2, 2)))
    _write_raster(tmp_path / 'tsf_los_20180101.tif', np.zeros((2, 2)))

    series = Sentinel1LOSLoader().load(tmp_path, 'tsf_los_*.tif')

    assert series.dates == (date(2018, 1, 1), date(2018, 1, 13))
    assert series.data.shape == (2, 2, 2)
    assert np.all(series.data[0] == 0)


def test_sentinel1_loader_rejects_invalid_filename(tmp_path):
    _write_raster(tmp_path / 'los_20180101.tif', np.ones((2, 2)))

    with pytest.raises(ValueError, match='Invalid LOS filename'):
        Sentinel1LOSLoader().load(tmp_path, '*.tif')


def test_apply_mask_sets_pixels_outside_tsf_to_nan():
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    mask = np.array([[1, 0], [1, 0]])

    masked = apply_mask(data, mask)

    assert np.isnan(masked[:, 0, 1]).all()
    assert np.all(masked[:, 1, 0] == data[:, 1, 0])


def test_time_series_statistics_uses_nested_schema():
    data = np.array(
        [
            [[1.0, 2.0], [np.nan, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ],
    )

    statistics = time_series_statistics(
        data,
        (date(2018, 1, 1), date(2018, 1, 13)),
        histogram_bins=3,
    )

    assert 'per_acquisition' in statistics
    assert 'overall' in statistics
    assert statistics['per_acquisition']['2018-01-01']['mean'] == pytest.approx(
        7.0 / 3.0,
    )
    assert len(statistics['overall']['global_histogram']['counts']) == 3


def test_slope_feature_extractor_computes_enabled_features():
    dates = (
        date(2018, 1, 1),
        date(2018, 1, 11),
        date(2018, 1, 21),
        date(2018, 1, 31),
    )
    data = np.array(
        [
            [[0.0, np.nan], [0.0, 0.0]],
            [[10.0, np.nan], [10.0, 10.0]],
            [[20.0, np.nan], [20.0, 20.0]],
            [[30.0, np.nan], [30.0, 30.0]],
        ],
        dtype=np.float32,
    )

    features = SlopeFeatureExtractor().compute(
        data,
        dates,
        {
            'cumulative_displacement': True,
            'velocity': True,
            'acceleration': True,
            'jerk': True,
            'minimum': True,
            'maximum': True,
            'mean': True,
            'standard_deviation': True,
            'variance': True,
            'range': True,
            'trend': True,
        },
    )

    assert features['cumulative_displacement'][0, 0] == pytest.approx(60.0)
    assert features['velocity'][0, 0] == pytest.approx(1.0)
    assert features['acceleration'][0, 0] == pytest.approx(0.0)
    assert features['jerk'][0, 0] == pytest.approx(0.0)
    assert features['minimum'][0, 0] == pytest.approx(0.0)
    assert features['maximum'][0, 0] == pytest.approx(30.0)
    assert features['mean'][0, 0] == pytest.approx(15.0)
    assert features['range'][0, 0] == pytest.approx(30.0)
    assert features['trend'][0, 0] == pytest.approx(1.0)
    assert np.isnan(features['cumulative_displacement'][0, 1])


def test_temporal_feature_extractor_computes_enabled_features():
    dates = tuple(date(2018, 1, day) for day in range(1, 8))
    stack = np.arange(28, dtype=np.float32).reshape(7, 2, 2)
    stack[:, 0, 1] = np.nan

    features = TemporalFeatureExtractor().compute(
        {'velocity': stack},
        dates,
        {
            'lag': {
                'enabled': True,
                'orders': [1, 2],
            },
            'difference': {
                'enabled': True,
                'orders': [1, 2],
            },
            'rolling_mean': {
                'enabled': True,
                'window': 3,
            },
            'rolling_std': {
                'enabled': True,
                'window': 3,
            },
            'smoothing': {
                'enabled': True,
                'method': 'savgol',
                'window': 5,
                'polyorder': 2,
            },
        },
    )

    assert features['velocity_lag1'].shape == stack.shape
    assert np.isnan(features['velocity_lag1'][0, 0, 0])
    assert features['velocity_lag1'][-1, 0, 0] == pytest.approx(20.0)
    assert features['velocity_lag2'][-1, 0, 0] == pytest.approx(16.0)
    assert features['velocity_diff1'][-1, 0, 0] == pytest.approx(4.0)
    assert features['velocity_diff2'][-1, 0, 0] == pytest.approx(8.0)
    assert np.isnan(features['velocity_roll_mean'][1, 0, 0])
    assert features['velocity_roll_mean'][-1, 0, 0] == pytest.approx(20.0)
    assert features['velocity_roll_std'][-1, 0, 0] == pytest.approx(
        np.std([16.0, 20.0, 24.0]),
    )
    # assert features['velocity_smooth'][-1, 0, 0] == pytest.approx(24.0)
    assert np.isnan(features['velocity_lag1'][-1, 0, 1])
