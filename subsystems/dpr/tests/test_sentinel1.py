import sys
import pytest
from pathlib import Path
import os
from shapely.wkt import loads
from shapely.geometry import Polygon
import numpy as np


# to be removed when https://github.com/GAIA-TSF/GAIA-TSF-System/issues/97 is solved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from lib.config import ProjectConfigReader
from eou.data_acquisition_gateway import DataAcquisitionGateway
from dpr.preprocessing_pipelines import PreprocessingPipelines


@pytest.fixture(scope='class')
def pipeline():
    module = PreprocessingPipelines()
    return module.pipeline['sentinel1']


@pytest.fixture(scope='class')
def config():
    return ProjectConfigReader(
        str(
            Path(__file__).parent.parent.parent.parent
            / 'tests'
            / 'projects'
            / 'sibanye-td6.yml'
        )
    )


class TestSentinel1Workflow:
    def _get_bbox(self, config):
        """Helper to extract bbox from config."""
        aoi = loads(config['project']['aoi']['geom'])
        min_lon, min_lat, max_lon, max_lat = aoi.bounds
        return Polygon([
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ])

    def test_config(self, config):
        """Test project configuration."""
        assert config.is_valid() is True

    def test_download(self, config):
        """Test EOU Data Acquisition Gateway to download Sentinel-1 data."""
        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2026-01-01',
            'end': '2026-01-29',
            'productType': 'S1_SAR_SLC',
            'orbitDirection': 'ascending',
        }

        module = DataAcquisitionGateway()
        results = module.search(
            geom=config['project']['aoi']['geom'],
            **search_filter,
        )
        assert len(results) > 0

        config_eodag = str(
            Path(__file__).parent.parent.parent / 'eou' / 'tests' / 'eodag_config.yml'
        )
        module.set_config(config_eodag)

        output_directory = config['project']['data_dir']
        data_path = None
        try:
            data_path = module.download(
                results[0], quicklook=False, output_dir=output_directory
            )
            assert Path(data_path).exists()
        finally:
            if data_path and Path(data_path).exists():
                Path(data_path).unlink()

    def test_001_download_orbits(self, pipeline, config):
        data_dir = config['project']['data_dir']
        pipeline._download_orbits(data_dir)
        eof_files = [f for f in os.listdir(data_dir) if f.endswith('.EOF')]
        assert len(eof_files) > 0

    def test_002_download_dem_baseline(self, pipeline, config):
        bbox = self._get_bbox(config)
        pipeline._download_dem_baseline(bbox)
        assert pipeline.dem_da is not None
        assert pipeline.dem_da.rio.crs.to_epsg() == 4326
        assert 'lat' in pipeline.dem_da.coords
        assert 'lon' in pipeline.dem_da.coords
        assert pipeline.dem_da.ndim == 2

    def test_003_lidar_infill(self, pipeline, config):
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))

        base_dir = Path(config['project']['data_dir'])
        lidar_dir = base_dir / 'lidar'
        if lidar_dir.is_dir():
            lidar_file = next(lidar_dir.glob('*lidar*.nc'), None)
            if lidar_file is None:
                raise FileNotFoundError(f'No lidar .nc file found in {lidar_dir}')
            else:
                baseline_snapshot = pipeline.dem_da.copy(deep=True)
                pipeline._lidar_infill(lidar_file)
                assert pipeline.dem_da is not None
                assert not np.array_equal(pipeline.dem_da.values, baseline_snapshot.values)
        else:
            assert pipeline.dem_da is not None

    def test_run_workflow(self, pipeline, config):
        pass
