"""Tests for GIS metadata emitted beside MAP GeoTIFF products."""

from __future__ import annotations

import json

import numpy as np
from rasterio.transform import from_origin

from subsystems.map.dataset.dataset_builder import Dataset
from subsystems.map.dataset.feature_loader import RasterGrid
from subsystems.map.monitoring.residual_analysis import ResidualAnalyzer


def test_map_raster_writer_creates_same_stem_stac_sidecar(tmp_path) -> None:
    dataset = Dataset(
        features=np.empty((0, 0)),
        targets=np.empty(0),
        time_indices=np.empty(0, dtype=np.int64),
        pixel_indices=np.empty(0, dtype=np.int64),
        feature_names=(),
        dates=('2020-01-01',),
        grid=RasterGrid(
            crs='EPSG:32633',
            transform=from_origin(500000, 6700000, 10, 10),
            height=2,
            width=2,
            nodata=np.nan,
        ),
        mask=np.ones((2, 2), dtype=bool),
    )
    output = tmp_path / 'residual_20200101.tif'

    ResidualAnalyzer._write_raster(
        output,
        np.array([[0.1, np.nan], [0.3, 0.4]]),
        dataset,
        'residual_rate',
        unit='m/day',
    )

    item = json.loads(output.with_suffix('.json').read_text())
    assert item['stac_version'] == '1.0.0'
    assert item['assets']['data']['href'] == './residual_20200101.tif'
    band = item['properties']['raster:bands'][0]
    assert band['name'] == 'residual_rate'
    assert band['unit'] == 'm/day'
    assert band['nodata'] is None
