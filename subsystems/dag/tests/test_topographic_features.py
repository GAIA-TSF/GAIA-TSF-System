from __future__ import annotations

import json

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

from subsystems.dag.pipelines.topographic_feature_pipeline import (
    TopographicFeaturePipeline,
)
from subsystems.dag.plugins.features.topographic_features import (
    TopographicFeatureExtractor,
)


def test_topographic_extractor_computes_slope_and_pi():
    # One metre of elevation gain per 10 metres in the x direction.
    dem = np.tile(np.arange(5, dtype=np.float32), (5, 1))
    features = TopographicFeatureExtractor().compute(dem, 10.0, 10.0, 3)

    assert set(features) == {'dem', 'slope', 'pi'}
    assert np.allclose(features['slope'], np.degrees(np.arctan(0.1)))
    assert np.isclose(features['pi'][2, 2], 0.0)


def test_topographic_pipeline_writes_static_feature_set(tmp_path):
    project_dir = tmp_path / 'project'
    dem_path = project_dir / 'static' / 'tsf_dem.tif'
    dem_path.parent.mkdir(parents=True)
    dem = np.array(
        [
            [100, 100, 100, 100, 100],
            [100, 101, 102, 101, 100],
            [100, 102, 110, 102, 100],
            [100, 101, 102, 101, 100],
            [100, 100, 100, 100, 100],
        ],
        dtype=np.float32,
    )
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=5,
        width=5,
        count=1,
        dtype='float32',
        crs='EPSG:32633',
        transform=from_origin(0, 50, 10, 10),
        nodata=np.nan,
    ) as dataset:
        dataset.write(dem, 1)

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'project_dir': str(project_dir),
                'static_topography': {
                    'dem': 'static/tsf_dem.tif',
                    'pi_window_size': 3,
                    'results': {
                        'output_dir': 'results/static_features',
                        'raster_format': 'GTiff',
                        'metadata_filename': 'metadata.json',
                        'filenames': {
                            'dem': 'dem.tif',
                            'slope': 'slope.tif',
                            'pi': 'pi.tif',
                        },
                    },
                },
            }
        ),
        encoding='utf-8',
    )

    result = TopographicFeaturePipeline(config_path).run()

    output_dir = project_dir / 'results' / 'static_features'
    assert result['features'] == ['dem', 'pi', 'slope']
    assert (output_dir / 'dem.tif').exists()
    assert (output_dir / 'slope.tif').exists()
    assert (output_dir / 'pi.tif').exists()
    with rasterio.open(output_dir / 'pi.tif') as dataset:
        pi = dataset.read(1)
        assert dataset.crs.to_epsg() == 32633
        assert np.isclose(pi[2, 2], 110 - np.mean(dem[1:4, 1:4]))
    metadata = json.loads((output_dir / 'metadata.json').read_text())
    assert metadata['units'] == {'dem': 'm', 'slope': 'degree', 'pi': 'm'}
    assert metadata['pi_window_size'] == 3
