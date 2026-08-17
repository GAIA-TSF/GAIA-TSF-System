from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml

from subsystems.dag.pipelines.slope_eda_pipeline import SlopeEDAPipeline
from subsystems.dag.pipelines.slope_feature_pipeline import SlopeFeaturePipeline
from subsystems.dag.pipelines.slope_temporal_feature_pipeline import (
    SlopeTemporalFeaturePipeline,
)


def _write_raster(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_slope_eda_pipeline_writes_statistics_and_maps(tmp_path):
    project_dir = tmp_path / 'project'
    input_dir = project_dir / 'inputs'
    static_dir = project_dir / 'static'
    output_dir = project_dir / 'results' / 'eda'
    _write_raster(input_dir / 'tsf_los_20180101.tif', np.array([[1, 2], [3, 4]]))
    _write_raster(input_dir / 'tsf_los_20180113.tif', np.array([[2, 4], [6, 8]]))
    _write_raster(static_dir / 'tsf_mask.tif', np.array([[1, 1], [0, 1]]))

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'project_dir': str(project_dir),
                'slope_stability': {
                    'inputs': {
                        'los': {
                            'directory': 'inputs',
                            'filename_pattern': 'tsf_los_*.tif',
                        },
                    },
                    'static': {
                        'tsf_mask': 'static/tsf_mask.tif',
                    },
                    'results': {
                        'eda': {
                            'output_dir': 'results/eda',
                            'histogram_bins': 4,
                            'dpi': 80,
                            'cmap': 'viridis',
                            'raster_format': 'GTiff',
                            'filenames': {
                                'statistics': 'statistics.json',
                                'temporal_mean_std': 'temporal_mean_std.png',
                                'histogram': 'histogram.png',
                                'boxplot': 'boxplot.png',
                                'mean_heatmap': 'mean_los_heatmap.png',
                                'std_heatmap': 'temporal_std_heatmap.png',
                                'mean_map': 'mean_map.tif',
                                'std_map': 'std_map.tif',
                            },
                        },
                    },
                },
            },
        ),
        encoding='utf-8',
    )

    result = SlopeEDAPipeline(config_path).run()

    assert result['acquisitions'] == 2
    statistics = json.loads((output_dir / 'statistics.json').read_text())
    assert statistics['overall']['overall_mean'] == 3.5
    assert (output_dir / 'temporal_mean_std.png').exists()
    assert (output_dir / 'histogram.png').exists()
    assert (output_dir / 'boxplot.png').exists()
    assert (output_dir / 'mean_los_heatmap.png').exists()
    assert (output_dir / 'temporal_std_heatmap.png').exists()
    assert (output_dir / 'mean_map.tif').exists()
    assert (output_dir / 'std_map.tif').exists()


def test_slope_feature_pipeline_writes_feature_rasters_and_metadata(tmp_path):
    project_dir = tmp_path / 'project'
    input_dir = project_dir / 'inputs'
    static_dir = project_dir / 'static'
    output_dir = project_dir / 'results' / 'features'

    _write_raster(input_dir / 'tsf_los_20180101.tif', np.array([[0, 0], [0, 0]]))
    _write_raster(input_dir / 'tsf_los_20180111.tif', np.array([[10, 10], [10, 10]]))
    _write_raster(input_dir / 'tsf_los_20180121.tif', np.array([[20, 20], [20, 20]]))
    _write_raster(input_dir / 'tsf_los_20180131.tif', np.array([[30, 30], [30, 30]]))
    _write_raster(static_dir / 'tsf_mask.tif', np.array([[1, 0], [1, 1]]))

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'project_dir': str(project_dir),
                'slope_stability': {
                    'feature_engineering': {
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
                        'temporal_variance': True,
                    },
                    'inputs': {
                        'los': {
                            'directory': 'inputs',
                            'filename_pattern': 'tsf_los_*.tif',
                        },
                    },
                    'static': {
                        'tsf_mask': 'static/tsf_mask.tif',
                    },
                    'results': {
                        'features': {
                            'output_dir': 'results/features',
                            'raster_format': 'GTiff',
                            'metadata_filename': 'metadata.json',
                            'filenames': {
                                'cumulative_displacement': (
                                    'cumulative_displacement.tif'
                                ),
                                'velocity': 'velocity.tif',
                                'acceleration': 'acceleration.tif',
                                'jerk': 'jerk.tif',
                                'minimum': 'minimum.tif',
                                'maximum': 'maximum.tif',
                                'mean': 'mean.tif',
                                'standard_deviation': 'std.tif',
                                'variance': 'variance.tif',
                                'range': 'range.tif',
                                'trend': 'trend.tif',
                                'temporal_variance': 'temporal_variance.tif',
                            },
                        },
                    },
                },
            },
        ),
        encoding='utf-8',
    )

    result = SlopeFeaturePipeline(config_path).run()

    assert result['pipeline'] == 'slope_features'
    assert (output_dir / 'velocity.tif').exists()
    assert (output_dir / 'cumulative_displacement.tif').exists()
    assert (output_dir / 'metadata.json').exists()

    metadata = json.loads((output_dir / 'metadata.json').read_text())
    assert 'velocity' in metadata['feature_names']
    assert metadata['statistics']['velocity']['mean'] == 1.0


def test_slope_temporal_feature_pipeline_writes_rasters_and_metadata(tmp_path):
    project_dir = tmp_path / 'project'
    input_dir = project_dir / 'inputs'
    static_dir = project_dir / 'static'
    output_dir = project_dir / 'results' / 'temporal_features'

    for index in range(7):
        value = float(index * 10)
        date_string = f'201801{index + 1:02d}'
        _write_raster(
            input_dir / f'tsf_los_{date_string}.tif',
            np.array([[value, value], [value, value]]),
        )
    _write_raster(static_dir / 'tsf_mask.tif', np.array([[1, 0], [1, 1]]))

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'project_dir': str(project_dir),
                'slope_stability': {
                    'inputs': {
                        'los': {
                            'directory': 'inputs',
                            'filename_pattern': 'tsf_los_*.tif',
                        },
                    },
                    'static': {
                        'tsf_mask': 'static/tsf_mask.tif',
                    },
                    'results': {
                        'temporal_features': {
                            'enabled': True,
                            'output_dir': 'results/temporal_features',
                            'raster_format': 'GTiff',
                            'metadata_filename': 'metadata.json',
                            'input_features': ['velocity'],
                            'lag': {
                                'enabled': True,
                                'orders': [1, 2],
                            },
                            'difference': {
                                'enabled': True,
                                'orders': [1],
                            },
                            'rolling_mean': {
                                'enabled': True,
                                'window': 3,
                            },
                            'rolling_std': {
                                'enabled': True,
                                'window': 3,
                            },
                            'calendar': {
                                'enabled': True,
                                'annual_period_days': 365.2425,
                                'features': ['annual_sin', 'annual_cos'],
                            },
                            'smoothing': {
                                'enabled': True,
                                'method': 'savgol',
                                'window': 5,
                                'polyorder': 2,
                            },
                            'filenames': {},
                        },
                    },
                },
            },
        ),
        encoding='utf-8',
    )

    result = SlopeTemporalFeaturePipeline(config_path).run()

    assert result['pipeline'] == 'slope_temporal_features'
    assert (output_dir / 'velocity_lag1.tif').exists()
    assert (output_dir / 'velocity_diff1.tif').exists()
    assert (output_dir / 'velocity_roll_mean.tif').exists()
    assert (output_dir / 'velocity_roll_std.tif').exists()
    assert (output_dir / 'annual_sin.tif').exists()
    assert (output_dir / 'annual_cos.tif').exists()
    # assert (output_dir / 'velocity_smooth.tif').exists()
    with rasterio.open(output_dir / 'velocity_lag1.tif') as dataset:
        assert dataset.count == 7
        assert dataset.descriptions == tuple(
            f'2018-01-{day:02d}' for day in range(1, 8)
        )

    metadata = json.loads((output_dir / 'metadata.json').read_text())
    assert metadata['base_feature_names'] == ['velocity']
    assert metadata['acquisition_dates'] == [
        f'2018-01-{day:02d}' for day in range(1, 8)
    ]
    assert 'velocity_lag1' in metadata['feature_names']
    assert 'annual_sin' in metadata['feature_names']
    with rasterio.open(output_dir / 'annual_sin.tif') as dataset:
        annual_sin = dataset.read(1, masked=True)
        assert np.isclose(annual_sin[0, 0], 0.0)
        assert annual_sin.mask[0, 1]
