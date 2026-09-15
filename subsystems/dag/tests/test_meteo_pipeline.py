from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

from subsystems.dag.pipelines.meteo_feature_pipeline import MeteoFeaturePipeline


def _write_raster(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=2,
        width=2,
        count=1,
        dtype='float32',
        crs='EPSG:32633',
        transform=from_origin(0, 20, 10, 10),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.full((2, 2), value, dtype=np.float32), 1)


def test_meteo_pipeline_writes_dated_feature_rasters(tmp_path):
    project = tmp_path / 'project'
    for day, precipitation in ((1, 2.0), (2, 20.0), (3, 4.0)):
        suffix = f'202501{day:02d}'
        _write_raster(
            project / 'inputs' / 'precip' / f'precip_{suffix}.tif',
            precipitation,
        )
        _write_raster(project / 'inputs' / 'mean' / f'mean_{suffix}.tif', 1.0)
        _write_raster(project / 'inputs' / 'min' / f'min_{suffix}.tif', -1.0)
        _write_raster(project / 'inputs' / 'max' / f'max_{suffix}.tif', 2.0)
        _write_raster(project / 'inputs' / 'insar' / f'los_{suffix}.tif', 0.0)
    _write_raster(project / 'static' / 'mask.tif', 1.0)

    config = {
        'project_dir': str(project),
        'meteorology': {
            'inputs': {
                'insar': {
                    'directory': 'inputs/insar',
                    'filename_pattern': 'los_*.tif',
                },
                'precipitation': {
                    'directory': 'inputs/precip',
                    'filename_pattern': 'precip_*.tif',
                },
                'temperature_mean': {
                    'directory': 'inputs/mean',
                    'filename_pattern': 'mean_*.tif',
                },
                'temperature_min': {
                    'directory': 'inputs/min',
                    'filename_pattern': 'min_*.tif',
                },
                'temperature_max': {
                    'directory': 'inputs/max',
                    'filename_pattern': 'max_*.tif',
                },
            },
            'static': {'tsf_mask': 'static/mask.tif'},
            'feature_engineering': {
                'precip_7d': True,
                'temp_7d_mean': True,
                'cold_regions': {'enabled': True},
            },
            'results': {
                'output_dir': 'results/meteo',
                'metadata_filename': 'metadata.json',
                'filenames': {},
            },
        },
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

    result = MeteoFeaturePipeline(config_path).run()

    assert result['pipeline'] == 'meteo_features'
    output = project / 'results' / 'meteo'
    with rasterio.open(output / 'precip_7d.tif') as dataset:
        assert dataset.count == 3
        assert dataset.descriptions == ('2025-01-01', '2025-01-02', '2025-01-03')
        assert dataset.read(3)[0, 0] == 26.0
    metadata = json.loads((output / 'metadata.json').read_text())
    assert 'freeze_thaw' in metadata['feature_names']
    assert metadata['dates'][-1] == '2025-01-03'


def test_meteo_pipeline_reads_daily_csv_input(tmp_path):
    project = tmp_path / 'project'
    _write_raster(project / 'static' / 'mask.tif', 1.0)
    _write_raster(project / 'inputs' / 'insar' / 'los_20250102.tif', 0.0)
    input_path = project / 'inputs' / 'meteodata.csv'
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        'date,precipitation,temperature_mean,temperature_min,temperature_max\n'
        '20250101,1.0,-1.0,-3.0,1.0\n'
        '20250102,4.0,2.0,0.0,4.0\n',
        encoding='utf-8',
    )
    config = {
        'project_dir': str(project),
        'meteorology': {
            'inputs': {
                'insar': {
                    'directory': 'inputs/insar',
                    'filename_pattern': 'los_*.tif',
                },
                'table': {
                    'path': 'inputs/meteodata.csv',
                    'date_column': 'date',
                    'columns': {},
                },
            },
            'static': {'tsf_mask': 'static/mask.tif'},
            'feature_engineering': {
                'precip_7d': True,
                'temperature_mean': True,
            },
            'results': {'output_dir': 'results/meteo', 'filenames': {}},
        },
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

    MeteoFeaturePipeline(config_path).run()

    output_path = project / 'results' / 'meteo' / 'precip_7d.tif'
    with rasterio.open(output_path) as dataset:
        assert dataset.count == 1
        assert dataset.descriptions == ('2025-01-02',)
        assert dataset.read(1)[0, 0] == 5.0
