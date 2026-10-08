"""Spatial calibration support for sites without a stable time period."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from subsystems.map.dataset.feature_loader import RasterGrid
from subsystems.map.pipelines.learning_pipeline import LearningPipeline


def _write_mask(path: Path, values: np.ndarray) -> RasterGrid:
    transform = from_origin(0, 20, 10, 10)
    with rasterio.open(
        path, 'w', driver='GTiff', width=2, height=2, count=1,
        dtype='uint8', crs='EPSG:32633', transform=transform, nodata=0,
    ) as target:
        target.write(values.astype('uint8'), 1)
    return RasterGrid(rasterio.crs.CRS.from_epsg(32633), transform, 2, 2, 0)


def _pipeline(path: Path, method: str) -> LearningPipeline:
    pipeline = LearningPipeline.__new__(LearningPipeline)
    pipeline.config = {'baseline_model': {'stable_pixel_std_threshold': 0.01}}
    pipeline._static_file = lambda filename: path
    pipeline._method = method
    return pipeline


def test_configured_stable_mask_controls_training_population(tmp_path):
    path = tmp_path / 'stable.tif'
    grid = _write_mask(path, np.array([[1, 0], [0, 1]]))
    target = np.array([
        [[0.0, 1.0], [2.0, 3.0]],
        [[0.5, 2.0], [4.0, 3.5]],
    ])
    loaded = SimpleNamespace(
        features={'target': target}, mask=np.ones((2, 2), dtype=bool), grid=grid,
        dates=('2020-01-01', '2020-01-02'),
    )
    mask, metadata = _pipeline(path, 'configured_mask')._calibration_mask(
        {'stable_file': 'stable.tif',
         'calibration_spatial_selection': {'method': 'configured_mask'}},
        loaded, 'target', 0, 2,
    )
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])
    assert metadata['selected_pixel_count'] == 2
    assert metadata['monitoring_domain'] == 'dataset.mask_file'


def test_stable_mask_must_be_subset_of_tsf(tmp_path):
    path = tmp_path / 'stable.tif'
    grid = _write_mask(path, np.array([[1, 1], [0, 0]]))
    loaded = SimpleNamespace(
        features={'target': np.ones((2, 2, 2))},
        mask=np.array([[1, 0], [0, 0]], dtype=bool), grid=grid,
        dates=('2020-01-01', '2020-01-02'),
    )
    with pytest.raises(ValueError, match='outside dataset.mask_file'):
        _pipeline(path, 'configured_mask')._calibration_mask(
            {'stable_file': 'stable.tif',
             'calibration_spatial_selection': {'method': 'configured_mask'}},
            loaded, 'target', 0, 2,
        )
