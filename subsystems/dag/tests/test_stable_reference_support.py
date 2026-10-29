"""External stable reference pixels must survive temporal feature masking."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from subsystems.dag.pipelines.slope_temporal_feature_pipeline import (
    SlopeTemporalFeaturePipeline,
)
from subsystems.dag.utils.raster import RasterProfile, apply_mask


class StableReferenceSupportTests(unittest.TestCase):
    def test_external_reference_survives_feature_mask_and_correction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'static').mkdir()
            transform = from_origin(0, 20, 10, 10)
            profile = RasterProfile(
                rasterio.crs.CRS.from_epsg(32633), transform, 2, 2, 'float32', None,
            )
            path = root / 'static/tsf_stable.tif'
            with rasterio.open(
                path, 'w', driver='GTiff', width=2, height=2, count=1,
                dtype='uint8', crs=profile.crs, transform=transform, nodata=0,
            ) as dst:
                dst.write(np.array([[0, 1], [0, 0]], dtype='uint8'), 1)
            pipeline = SlopeTemporalFeaturePipeline.__new__(SlopeTemporalFeaturePipeline)
            pipeline._resolve_path = lambda filename: root / filename
            tsf = np.array([[1, 0], [0, 0]], dtype=bool)
            support = pipeline._include_stable_reference(tsf, {}, profile)
            np.testing.assert_array_equal(support, [[True, True], [False, False]])
            data = apply_mask(np.array([[[5., 2.], [7., 9.]]]), support)
            diagnostics, corrected = pipeline._stable_reference(
                data, {'stable_reference': {'common_mode_correction': True}},
            )
            assert diagnostics['per_date_median'] == [2.0]
            np.testing.assert_allclose(corrected[0, 0], [3., 0.])
            assert np.isnan(corrected[0, 1]).all()
            np.testing.assert_array_equal(tsf, [[True, False], [False, False]])
            shifted = RasterProfile(profile.crs, from_origin(1, 20, 10, 10), 2, 2, 'float32', None)
            with self.assertRaisesRegex(ValueError, 'grid does not match'):
                pipeline._include_stable_reference(tsf, {}, shifted)

    def test_missing_optional_mask_preserves_tsf(self):
        pipeline = SlopeTemporalFeaturePipeline.__new__(SlopeTemporalFeaturePipeline)
        with TemporaryDirectory() as directory:
            pipeline._resolve_path = lambda filename: Path(directory) / filename
            tsf = np.array([[True, False]])
            assert pipeline._include_stable_reference(tsf, {}, None) is tsf
