"""AMD formula and raster-to-point integration tests."""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import Point

from subsystems.dag.pipelines.amd_eda_pipeline import AMDEDAPipeline
from subsystems.dag.plugins.features.amd_features import calculate_amd_index


def test_formulas_preserve_gaps_and_reject_zero_denominators():
    first = np.array([6, 6, np.nan, 2])
    second = np.array([2, 0, 1, 0.01])
    np.testing.assert_allclose(
        calculate_amd_index(first, second, 'ratio', 0.1),
        [3, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        calculate_amd_index(first, second, 'difference'),
        [4, 6, np.nan, 1.99],
        equal_nan=True,
    )
    with pytest.raises(ValueError):
        calculate_amd_index(first, second, 'unknown')


@pytest.fixture
def scenario(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    static = tmp_path / 'static'
    static.mkdir()
    profile = {
        'driver': 'GTiff', 'width': 2, 'height': 1, 'crs': 'EPSG:32633',
        'transform': from_origin(0, 10, 10, 10), 'dtype': 'float32', 'nodata': -9999.0,
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


@pytest.mark.parametrize(
    'method, expected', [('ratio', [3, 5, 7]), ('difference', [4, 8, 12])]
)
def test_index_is_engineered_before_regional_and_point_eda(scenario, method, expected):
    path, config = scenario
    config['amd']['feature_engineering']['method'] = method
    path.write_text(yaml.safe_dump(config))
    result = AMDEDAPipeline(path).run()
    with rasterio.open(result['index']) as source:
        np.testing.assert_allclose(source.read()[:, 0, 0], expected)
        assert np.isnan(source.read(2)[0, 1])
        assert source.descriptions == ('2020-01-01', '2020-01-11', '2020-01-21')
    output = path.parent / 'results/eda'
    stats = json.loads((output / 'statistics.json').read_text())
    assert (
        stats['regions']['clean_water']['per_acquisition']['2020-01-11']['mean'] is None
    )
    quality = json.loads((output / 'quality_report.json').read_text())
    assert quality['errors'] == []
    assert quality['gap_days'] == [10, 10]
    assert quality['acquisitions'][1]['regions']['clean_water']['valid_fraction'] == 0
    assert (output / 'point_timeseries.png').stat().st_size > 0
    assert '2020-01-11' in (output / 'point_timeseries.csv').read_text()
    gaps = json.loads((output / 'gap_analysis.json').read_text())
    assert gaps['candidate_acquisitions'] == 3
    assert gaps['series']['model_domain']['valid_observations'] == 3
    assert gaps['series']['region:amd']['valid_fraction'] == 1
    assert gaps['series']['region:clean_water']['valid_observations'] == 2
    assert gaps['series']['region:clean_water']['valid_gap_days']['max'] == 20
    assert gaps['series']['region:clean_water'][
        'missing_acquisitions_between_valid'
    ]['max'] == 1
    assert (output / 'gap_intervals.csv').exists()
    assert (output / 'gap_coverage.png').stat().st_size > 0


def test_cloud_edge_buffer_adds_filtered_point_series(scenario):
    path, config = scenario
    config['amd']['quality']['cloud_edge_buffer_pixels'] = 1
    path.write_text(yaml.safe_dump(config))
    result = AMDEDAPipeline(path).run()
    with rasterio.open(result['cloud_edge_filtered_index']) as source:
        assert np.isnan(source.read(2)).all()
        assert np.isfinite(source.read(1)).all()
    rows = list(__import__('csv').DictReader(
        (path.parent / 'results/eda/point_timeseries.csv').open()
    ))
    removed = [row for row in rows if row['cloud_edge_removed'] == 'True']
    assert len(removed) == 1  # AMD point was valid before buffering; clean point was cloud.
    gaps = json.loads((path.parent / 'results/eda/gap_analysis.json').read_text())
    assert gaps['cloud_edge_contamination']['point_observations_removed'] == 1
    assert gaps['cloud_edge_contamination']['buffer_pixels'] == 1


def test_acquisition_cloud_filter_removes_entire_cloudy_image(scenario):
    path, config = scenario
    config['amd']['quality']['acquisition_cloud_filter'] = {
        'enabled': True,
        'maximum_cloudy_pixels': 0,
        'cloudy_scl_classes': [9],
    }
    path.write_text(yaml.safe_dump(config))
    result = AMDEDAPipeline(path).run()
    with rasterio.open(result['index']) as source:
        assert source.descriptions == ('2020-01-01', '2020-01-21')
    quality = json.loads((path.parent / 'results/eda/quality_report.json').read_text())
    selection = quality['acquisition_cloud_filter']
    assert selection['accepted_acquisitions'] == 2
    assert selection['rejected_acquisitions'] == 1
    assert selection['rejected_dates'] == ['2020-01-11']


def test_missing_metadata_still_writes_inventory_and_quality(scenario):
    path, _ = scenario
    next((path.parent / 'inputs').glob('*.json')).unlink()
    with pytest.raises(ValueError, match='inventory'):
        AMDEDAPipeline(path).run()
    assert (path.parent / 'results/eda/inventory.csv').exists()
    assert json.loads((path.parent / 'results/eda/quality_report.json').read_text())[
        'errors'
    ]


def test_grid_mismatch_is_reported(scenario):
    path, _ = scenario
    with rasterio.open(path.parent / 'static/amd_mask.tif', 'r+') as dest:
        dest.transform = from_origin(1, 10, 10, 10)
    with pytest.raises(ValueError, match='grid mismatch'):
        AMDEDAPipeline(path).run()
    assert json.loads((path.parent / 'results/eda/quality_report.json').read_text())[
        'errors'
    ]


def test_larger_aligned_images_are_cropped_without_resampling(scenario):
    path, _ = scenario
    for image in (path.parent / 'inputs').glob('*.tif'):
        with rasterio.open(image) as source:
            data, profile = source.read(), source.profile
        profile.update(width=4, height=3, transform=from_origin(-10, 20, 10, 10))
        padded = np.full((3, 3, 4), -9999., dtype='float32')
        padded[:, 1:2, 1:3] = data
        with rasterio.open(image, 'w', **profile) as dest:
            dest.write(padded)
    result = AMDEDAPipeline(path).run()
    with rasterio.open(result['index']) as source:
        assert source.shape == (1, 2)
        np.testing.assert_allclose(source.read()[:, 0, 0], [3, 5, 7])


def test_duplicate_acquisitions_are_rejected(scenario):
    path, _ = scenario
    image = next((path.parent / 'inputs').glob('*.tif'))
    duplicate = image.with_name(image.name.replace('S2A_', 'S2B_'))
    duplicate.write_bytes(image.read_bytes())
    duplicate.with_suffix('.json').write_bytes(image.with_suffix('.json').read_bytes())
    with pytest.raises(ValueError, match='inventory'):
        AMDEDAPipeline(path).run()


def test_missing_band_mapping_is_rejected(scenario):
    path, config = scenario
    config['amd']['inputs']['sentinel2']['band_positions'] = {}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='mapping is ambiguous'):
        AMDEDAPipeline(path).run()


def test_configured_scaling_applies_before_difference(scenario):
    path, config = scenario
    config['amd']['feature_engineering'].update(method='difference', scale=0.1, offset=-1)
    path.write_text(yaml.safe_dump(config))
    result = AMDEDAPipeline(path).run()
    with rasterio.open(result['index']) as source:
        np.testing.assert_allclose(source.read()[:, 0, 0], [0.4, 0.8, 1.2])
