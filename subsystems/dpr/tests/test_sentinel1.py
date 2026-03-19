import pytest
from pathlib import Path
import os
from shapely.wkt import loads
from shapely.geometry import Polygon
import numpy as np
import re

from lib.config import ProjectConfigReader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines


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
    def _get_bbox(self, config):
        """Helper to extract bbox from config."""
        aoi = loads(config['project']['aoi']['geom'])
        min_lon, min_lat, max_lon, max_lat = aoi.bounds
        return Polygon(
            [
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
                (min_lon, min_lat),
            ]
        )

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

    def test_001_search_bursts(self, pipeline, config):
        aoi = loads(config['project']['aoi']['geom'])
        pipeline._search_bursts(aoi, '2022-07-01', '2022-10-30', 'A')
        assert pipeline.bursts is not None
        assert len(pipeline.bursts) > 0

    def test_002_download_bursts(self, pipeline, config):
        if pipeline.bursts is None:
            aoi = loads(config['project']['aoi']['geom'])
            pipeline._search_bursts(aoi, '2022-07-01', '2022-10-30', 'A')
            assert pipeline.bursts is not None
            assert len(pipeline.bursts) > 0

        data_dir = config['project']['data_dir']
        pipeline._download_bursts('username', 'password', data_dir)
        assert any(
            item.is_dir() and 'IW' in item.name for item in Path(data_dir).iterdir()
        )

    def test_003_download_orbits(self, pipeline, config):
        data_dir = config['project']['data_dir']
        pipeline._download_orbits(data_dir)
        assert pipeline.s1 is not None
        assert not pipeline.s1.df.empty
        eof_files = [f for f in os.listdir(data_dir) if f.endswith('.EOF')]
        assert len(eof_files) > 0

    def test_004_download_dem_baseline(self, pipeline, config):
        bbox = self._get_bbox(config)
        pipeline._download_dem_baseline(bbox)
        assert pipeline.dem_da is not None
        assert pipeline.dem_da.rio.crs.to_epsg() == 4326
        assert 'lat' in pipeline.dem_da.coords
        assert 'lon' in pipeline.dem_da.coords
        assert pipeline.dem_da.ndim == 2

    def test_005_lidar_infill(self, pipeline, config):
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))
            assert pipeline.dem_da is not None

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
                assert not np.array_equal(
                    pipeline.dem_da.values, baseline_snapshot.values
                )
        else:
            assert pipeline.dem_da is not None

    def test_006_save_composite_dem(self, pipeline, config):
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))
            assert pipeline.dem_da is not None

        base_dir = Path(config['project']['data_dir'])
        output_dem = base_dir / 'dem.nc'
        pipeline._save_composite_dem(output_dem)
        assert output_dem.exists()

    def test_007_clip_dem(self, pipeline, config):
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))
            assert pipeline.dem_da is not None

        aoi = loads(config['project']['aoi']['geom'])
        pipeline._clip_dem(aoi)
        assert pipeline.dem_masked is not None
        assert pipeline.dem_cropped is not None

        # Check if AOI != BBOX, case of sibanye-td6
        if len(aoi.exterior.coords) > 5:
            assert pipeline.dem_cropped.size < pipeline.dem_masked.size

        assert not np.isnan(pipeline.dem_masked.values).all()
        assert not np.isnan(pipeline.dem_cropped.values).all()

    def test_008_save_landmask(self, pipeline, config):
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))
            assert pipeline.dem_da is not None
        if pipeline.dem_masked is None:
            aoi = loads(config['project']['aoi']['geom'])
            pipeline._clip_dem(aoi)
            assert pipeline.dem_masked is not None

        base_dir = Path(config['project']['data_dir'])
        output_landmask = base_dir / 'landmask.nc'
        pipeline._save_landmask(output_landmask)
        assert output_landmask.exists()

    def test_009_link_s1_with_dem(self, pipeline, config):
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))
            assert pipeline.dem_da is not None
        base_dir = Path(config['project']['data_dir'])
        dem = base_dir / 'dem.nc'
        if not dem.exists():
            pipeline._save_composite_dem(dem)
            assert dem.exists()

        pipeline._link_s1_with_dem(base_dir, dem)
        assert pipeline.s1 is not None
        assert pipeline.s1.DEM is not None
        assert pipeline.s1.DEM == str(dem)

    def test_010_infer_ref_date(self, pipeline, config):
        if pipeline.bursts is None:
            aoi = loads(config['project']['aoi']['geom'])
            pipeline._search_bursts(aoi, '2022-07-01', '2022-10-30', 'A')
            assert pipeline.bursts is not None
            assert len(pipeline.bursts) > 0

        pipeline._infer_ref_date()
        assert pipeline.ref_date is not None
        assert isinstance(pipeline.ref_date, str)
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', pipeline.ref_date)

    def test_run_workflow(self, pipeline, config):
        pass
