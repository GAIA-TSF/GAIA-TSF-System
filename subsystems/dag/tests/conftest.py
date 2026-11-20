"""Shared fixtures for DAG subsystem tests."""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import Point


@pytest.fixture
def scenario(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    static = tmp_path / 'static'
    static.mkdir()
    profile = {
        'driver': 'GTiff',
        'width': 2,
        'height': 1,
        'crs': 'EPSG:32633',
        'transform': from_origin(0, 10, 10, 10),
        'dtype': 'float32',
        'nodata': -9999.0,
    }
    for day, first in ((1, 6), (11, 10), (21, 14)):
        stem = f'S2A_MSIL2A_202001{day:02d}T102131_N0500_R065_T33VVG_20230707T093356'
        with rasterio.open(inputs / f'{stem}.tif', 'w', count=3, **profile) as dest:
            dest.write(
                np.array(
                    [[[first, 4]], [[2, 2]], [[6, 9 if day == 11 else 6]]],
                    dtype='float32',
                )
            )
        (inputs / f'{stem}.json').write_text(
            json.dumps({'properties': {'datetime': f'2020-01-{day:02d}T10:21:31Z'}})
        )
    for region, values, x in (('amd', [1, 0], 5), ('clean_water', [0, 1], 15)):
        with rasterio.open(
            static / f'{region}_mask.tif', 'w', count=1, **profile
        ) as dest:
            dest.write(np.array([values], dtype='float32'), 1)
        gpd.GeoDataFrame(
            {'label': [region]}, geometry=[Point(x, 5)], crs=profile['crs']
        ).to_file(static / f'{region}_point.gpkg', driver='GPKG')
    config = {
        'project_dir': str(tmp_path),
        'amd': {
            'inputs': {
                'sentinel2': {
                    'directory': 'inputs',
                    'filename_pattern': '*.tif',
                    'band_positions': {'B04': 1, 'B03': 2, 'SCL': 3},
                }
            },
            'static': {
                f'{region}_{kind}': f'static/{region}_{kind}.{ext}'
                for region in ('amd', 'clean_water')
                for kind, ext in (('mask', 'tif'), ('point', 'gpkg'))
            },
            'feature_engineering': {
                'first_band': 'B04',
                'second_band': 'B03',
                'method': 'ratio',
            },
            'quality': {'scl_band': 'SCL', 'excluded_scl_classes': [9]},
            'results': {
                'features': {'output_dir': 'results/features'},
                'eda': {'output_dir': 'results/eda'},
            },
        },
    }
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, config
