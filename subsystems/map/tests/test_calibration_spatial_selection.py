"""Reference (normal) training population selection by spatial mask."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from conftest import project_config, write_project, write_stack
from subsystems.map.dataset.feature_loader import RasterGrid
from subsystems.map.pipelines.learning_pipeline import LearningPipeline, run_learning


def _write_mask(path: Path, values: np.ndarray) -> RasterGrid:
    transform = from_origin(0, 20, 10, 10)
    with rasterio.open(
        path, 'w', driver='GTiff', width=2, height=2, count=1, dtype='uint8',
        crs='EPSG:32633', transform=transform, nodata=0,
    ) as target:
        target.write(values.astype('uint8'), 1)
    return RasterGrid(rasterio.crs.CRS.from_epsg(32633), transform, 2, 2, 0)


def _pipeline(path: Path) -> LearningPipeline:
    pipeline = LearningPipeline.__new__(LearningPipeline)
    pipeline.ctx = SimpleNamespace(static_file=lambda filename: path)
    return pipeline


def _selection(method: str, threshold: float | None = None) -> dict:
    return {
        'method': method,
        'uses_std': method != 'configured_mask',
        'std_feature': 'velocity',
        'std_threshold': threshold,
        'mask_file': 'stable.tif',
    }


def test_configured_stable_mask_controls_training_population(tmp_path):
    path = tmp_path / 'stable.tif'
    grid = _write_mask(path, np.array([[1, 0], [0, 1]]))
    loaded = SimpleNamespace(
        features={'velocity': np.zeros((2, 2, 2))},
        mask=np.ones((2, 2), dtype=bool),
        grid=grid,
        dates=('2020-01-01', '2020-01-02'),
    )
    mask, metadata = _pipeline(path)._calibration_mask(
        _selection('configured_mask'), loaded, 0, 2
    )
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])
    assert metadata['selected_pixel_count'] == 2
    assert metadata['std_threshold'] is None


def test_temporal_std_uses_configured_feature_and_threshold(tmp_path):
    path = tmp_path / 'stable.tif'
    grid = _write_mask(path, np.array([[1, 1], [1, 0]]))
    velocity = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.001, 1.0], [0.0, 0.0]]])
    loaded = SimpleNamespace(
        features={'velocity': velocity},
        mask=np.ones((2, 2), dtype=bool),
        grid=grid,
        dates=('2020-01-01', '2020-01-02'),
    )
    mask, metadata = _pipeline(path)._calibration_mask(
        _selection('configured_mask_and_temporal_std', 0.01), loaded, 0, 2
    )
    np.testing.assert_array_equal(mask, [[True, False], [True, False]])
    assert metadata['std_feature'] == 'velocity'


def test_stable_pixels_outside_monitoring_mask_train_the_model(tmp_path):
    """README: the stable area may lie outside mask_file and still train."""
    root = write_project(tmp_path / 'p', 'slope')
    tsf = np.zeros((4, 5))
    tsf[:2] = 1  # monitoring domain: rows 0-1; stable_file: rows 2-3
    write_stack(root / 'static' / 'tsf_mask.tif', tsf, ())
    metadata = run_learning(project_config(root, 'slope_dataset', 'constant'))
    selection = metadata['reference_selection']
    assert selection['method'] == 'configured_mask'
    assert selection['selected_pixel_count'] == 10
    assert metadata['split_date_ranges']['train']['sample_count'] > 0
    with pytest.raises(FileNotFoundError):
        _pipeline(tmp_path / 'missing.tif')._calibration_mask(
            _selection('configured_mask'), SimpleNamespace(mask=None), 0, 1
        )
