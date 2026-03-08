import sys
import pytest
from pathlib import Path
import os
from shapely.wkt import loads
from shapely.geometry import Polygon


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
            / 'jagersfontein.yml'
        )
    )


class TestSentinel1Workflow:
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
        aoi = loads(config['project']['aoi']['geom'])
        min_lon, min_lat, max_lon, max_lat = aoi.bounds
        bbox = Polygon([
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ])

        result = pipeline._download_dem_baseline(bbox)
        assert result.rio.crs.to_epsg() == 4326
        assert "lat" in result.coords
        assert "lon" in result.coords
        assert result.ndim == 2

    def test_run_workflow(self, pipeline, config):
        pass
