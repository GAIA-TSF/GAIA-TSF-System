import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import Point

SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'extract_synthetic_tsf_in-situ_deformations.py'
)
SPEC = importlib.util.spec_from_file_location('extract_synthetic_insitu', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
COMPARE_SCRIPT = Path(__file__).parents[1] / 'scripts' / 'compare_insar_insitu.py'
COMPARE_SPEC = importlib.util.spec_from_file_location(
    'compare_insar_insitu', COMPARE_SCRIPT
)
COMPARE_MODULE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE_MODULE)


def test_extracts_all_geopackage_points_with_window_mean(tmp_path):
    project = tmp_path / 'project'
    true_los = project / 'inputs' / 'true_los'
    true_los.mkdir(parents=True)
    raster = np.arange(25, dtype='float32').reshape(5, 5) / 1000
    with rasterio.open(
        true_los / 'TRUE_LOS_20180101.tif',
        'w',
        driver='GTiff',
        width=5,
        height=5,
        count=1,
        dtype='float32',
        crs='EPSG:32633',
        transform=from_origin(0, 50, 10, 10),
    ) as dst:
        dst.write(raster, 1)
    los = project / 'inputs' / 'los'
    los.mkdir()
    insar_raster = np.zeros((5, 5), dtype='float32')
    insar_raster[1, 1] = 0.09
    insar_raster[3, 3] = 0.18
    with rasterio.open(
        los / 'tsf_los_20180101.tif',
        'w',
        driver='GTiff',
        width=5,
        height=5,
        count=1,
        dtype='float32',
        crs='EPSG:32633',
        transform=from_origin(0, 50, 10, 10),
    ) as dst:
        dst.write(insar_raster, 1)

    static = project / 'static'
    static.mkdir()
    gpd.GeoDataFrame(
        {'lable': ['stable_tsf', 'deformation_zone']},
        geometry=[Point(15, 35), Point(35, 15)],
        crs='EPSG:32633',
    ).to_file(static / 'observation_points.gpkg')
    config = {
        'project_dir': str(project),
        'global': {'random_seed': 42},
        'in_situ': {
            'inputs': {'directory': 'inputs/true_los', 'filename_pattern': '*.tif'},
            'static': {
                'observation_points': 'static/observation_points.gpkg',
                'label_column': 'lable',
            },
            'sampling': {'window_size': 3, 'sensor_noise_std_mm': 0},
            'output_csv': 'inputs/in_situ_deformation.csv',
            'platform': 'OFFICE_REVIEW',
            'qc': 0,
            'validation': {
                'insar_directory': 'inputs/los',
                'filename_pattern': '*.tif',
                'sampling_window_size': 3,
                'unit_scale': 1000,
                'output_csv': 'results/validation/comparison.csv',
                'statistics_json': 'results/validation/statistics.json',
            },
        },
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

    result = MODULE.extract(config_path)

    assert list(result.columns) == [
        'PLATFORM',
        'DATE',
        'LATITUDE',
        'LONGITUDE',
        'LOS_DEFORMATION',
        'QC',
    ]
    assert result['PLATFORM'].tolist() == ['OFFICE_REVIEW', 'OFFICE_REVIEW']
    assert result['QC'].tolist() == [0, 0]
    assert result['LOS_DEFORMATION'].tolist() == [6.0, 18.0]
    assert (project / 'inputs' / 'in_situ_deformation.csv').exists()
    assert not (project / 'results' / 'validation' / 'comparison.csv').exists()

    COMPARE_MODULE.compare(config_path)

    comparison = pd.read_csv(project / 'results' / 'validation' / 'comparison.csv')
    assert list(comparison.columns) == ['insar_los', 'insitu_deformation']
    # The centre pixels are 90 and 180 mm; the exported values prove that the
    # comparison uses the configured 3x3 mean (9 pixels), not the centre pixel.
    assert np.allclose(comparison['insar_los'], [10.0, 20.0])
    assert np.allclose(comparison['insitu_deformation'], [6.0, 18.0])
    statistics = json.loads(
        (project / 'results' / 'validation' / 'statistics.json').read_text()
    )
    assert statistics['sample_count'] == 2
    assert np.isclose(statistics['mean_absolute_error'], 3.0)
    assert np.isclose(
        statistics['root_mean_squared_error'],
        np.sqrt((4**2 + 2**2) / 2),
    )


def test_window_size_must_be_odd():
    try:
        MODULE.sample_neighbourhood(None, 0, 0, 2)
    except ValueError as error:
        assert 'positive odd' in str(error)
    else:
        raise AssertionError('even window size was accepted')


def test_loads_every_point_using_configured_label_column(tmp_path):
    points_file = tmp_path / 'observation_points.gpkg'
    gpd.GeoDataFrame(
        {'lable': ['stable_tsf', 'deformation_zone']},
        geometry=[Point(15, 35), Point(35, 15)],
        crs='EPSG:32633',
    ).to_file(points_file)
    assert MODULE.load_observation_points(points_file, 'EPSG:32633', 'lable') == {
        'stable_tsf': (15.0, 35.0),
        'deformation_zone': (35.0, 15.0),
    }
