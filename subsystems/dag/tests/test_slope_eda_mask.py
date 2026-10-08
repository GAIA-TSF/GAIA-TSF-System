from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from subsystems.dag.pipelines.slope_eda_pipeline import SlopeEDAPipeline
from subsystems.dag.utils.raster import RasterProfile, apply_mask
from subsystems.dag.utils.statistics import time_series_statistics


def test_eda_statistics_exclude_mask_background_and_nodata(tmp_path):
    path = tmp_path / 'tsf_mask.tif'
    transform = from_origin(0, 20, 10, 10)
    with rasterio.open(
        path, 'w', driver='GTiff', width=2, height=2, count=1,
        dtype='float32', crs='EPSG:32633', transform=transform, nodata=-9999,
    ) as source:
        source.write(np.array([[1, 0], [-9999, np.nan]], dtype='float32'), 1)
    profile = RasterProfile('EPSG:32633', transform, 2, 2, 'float32', -9999)
    pipeline = SlopeEDAPipeline.__new__(SlopeEDAPipeline)
    mask = pipeline._load_mask(path, (2, 2), profile)
    data = np.array([[[2, 100], [200, 300]], [[4, 400], [500, 600]]])
    statistics = time_series_statistics(
        apply_mask(data, mask), (date(2020, 1, 1), date(2020, 1, 2)), 2,
    )
    assert statistics['overall']['overall_mean'] == 3
    assert statistics['overall']['global_max'] == 4
    assert sum(statistics['overall']['global_histogram']['counts']) == 2
    assert statistics['per_acquisition']['2020-01-01']['mean'] == 2

    shifted = RasterProfile(
        profile.crs, from_origin(10, 20, 10, 10), 2, 2, 'float32', -9999,
    )
    with pytest.raises(ValueError, match='grid does not match'):
        pipeline._load_mask(path, (2, 2), shifted)
