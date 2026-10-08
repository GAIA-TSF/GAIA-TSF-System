"""Verify cloud-aware history and weather alignment for AMD."""

import json
from datetime import date, timedelta

import numpy as np
import pytest
import rasterio
import yaml

from subsystems.dag.tests.test_amd_eda import scenario  # noqa: F401
from subsystems.dag.pipelines.amd_model_feature_pipeline import (
    AMDFeaturePipeline, AMDTemporalFeaturePipeline, AMDMeteoFeaturePipeline,
)
from subsystems.dag.plugins.features.amd_temporal_features import compute_amd_temporal


def test_history_skips_clouds_and_crosses_years():
    data = np.array([1, np.nan, 3, 5, np.nan, 9], dtype=np.float32)[:, None, None]
    dates = tuple(date(2020, 12, 29) + timedelta(days=i) for i in range(6))
    features = compute_amd_temporal(data, dates, {'rolling_window': 3})
    np.testing.assert_allclose(features['lag1'][:, 0, 0],
                               [np.nan, np.nan, 1, 3, np.nan, 5], equal_nan=True)
    assert features['lag3'][5, 0, 0] == 1
    assert features['amd_diff2'][5, 0, 0] == 6
    assert features['roll_mean'][5, 0, 0] == pytest.approx(17 / 3)
    assert features['roll_std'][5, 0, 0] == pytest.approx(np.std([3, 5, 9]))
    assert np.isnan(features['roll_mean'][2, 0, 0])
    assert np.isnan(features['annual_sin'][1, 0, 0])


def test_shared_axis_and_weather_includes_cloudy_days(scenario):
    path, config = scenario
    amd = config['amd']
    amd['feature_engineering']['method'] = 'difference'
    amd['results']['temporal_features'] = {'output_dir': 'results/temporal'}
    amd['results']['meteorology'] = {'output_dir': 'results/weather'}
    amd['temporal_features'] = {'rolling_window': 2}
    amd['meteorology'] = {
        'inputs': {'table': {'path': 'weather.csv'}},
        'feature_engineering': {'precip_7d': True},
    }
    # Fully cloudy middle acquisition must disappear from the common date axis.
    image = sorted((path.parent / 'inputs').glob('*.tif'))[1]
    with rasterio.open(image, 'r+') as source:
        source.write(np.full((1, 2), 9, dtype=np.float32), 3)
    weather = 'date,precipitation\n' + ''.join(
        f'{date(2019, 12, 1) + timedelta(days=i)},1\n' for i in range(60)
    )
    (path.parent / 'weather.csv').write_text(weather)
    path.write_text(yaml.safe_dump(config))
    base = AMDFeaturePipeline(path).run()
    temporal = AMDTemporalFeaturePipeline(path).run()
    meteo = AMDMeteoFeaturePipeline(path).run()
    for result in (base, temporal, meteo):
        metadata = json.loads(open(result['metadata']).read())
        assert metadata['acquisition_dates'] == ['2020-01-01', '2020-01-21']
        for filename in result['output_files'].values():
            with rasterio.open(filename) as source:
                assert source.count == 2
                assert source.descriptions == ('2020-01-01', '2020-01-21')
    with rasterio.open(temporal['output_files']['amd_diff1']) as source:
        assert source.read(2)[0, 0] == 8
    with rasterio.open(meteo['output_files']['precip_7d']) as source:
        np.testing.assert_allclose(source.read(), 7)
    amd['acquisitions'] = {'min_valid_fraction': 0.9}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='rerun amd_features'):
        AMDTemporalFeaturePipeline(path).run()


def test_coverage_threshold_and_pixel_specific_lags(scenario):
    path, config = scenario
    config['amd']['acquisitions'] = {'min_valid_fraction': 0.75}
    path.write_text(yaml.safe_dump(config))
    result = AMDFeaturePipeline(path).run()
    metadata = json.loads(open(result['metadata']).read())
    assert metadata['excluded_dates'] == ['2020-01-11']
    data = np.array([[[1, 10]], [[np.nan, 20]], [[3, np.nan]], [[5, 40]]], dtype=np.float32)
    dates = tuple(date(2020, 1, 1) + timedelta(days=i * 20) for i in range(4))
    features = compute_amd_temporal(data, dates, {'rolling_window': 2})
    np.testing.assert_allclose(features['lag2'][3], [[1, 10]])
    np.testing.assert_allclose(features['roll_mean'][3], [[4, 30]])


def test_weather_preserves_partial_cloud_mask_and_baseline_required(scenario):
    path, config = scenario
    config['amd']['results']['meteorology'] = {'output_dir': 'results/weather'}
    config['amd']['meteorology'] = {
        'inputs': {'table': {'path': 'weather.csv'}},
        'feature_engineering': {'precipitation': True},
    }
    (path.parent / 'weather.csv').write_text('date,precipitation\n' + ''.join(
        f'2020-01-{i:02},2\n' for i in range(1, 22)))
    path.write_text(yaml.safe_dump(config))
    AMDFeaturePipeline(path).run()
    result = AMDMeteoFeaturePipeline(path).run()
    with rasterio.open(result['output_files']['precipitation']) as source:
        assert np.isnan(source.read(2)[0, 1])
        assert source.read(2)[0, 0] == 2
    config['amd']['meteorology']['feature_engineering']['temperature_anomaly'] = True
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='temperature_baseline'):
        AMDMeteoFeaturePipeline(path).run()
