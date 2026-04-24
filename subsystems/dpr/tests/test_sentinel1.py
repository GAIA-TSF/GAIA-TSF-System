from pathlib import Path
import os

from shapely.wkt import loads
from shapely.geometry import Polygon
import pytest
import xarray as xr
from dask.distributed import Client
import geopandas as gpd

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

    def test_download_sentinel1_slc(self, config):
        """Test EOU Data Acquisition Gateway to download Sentinel-1 data."""
        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2022-01-01',
            'end': '2022-01-31',
            'productType': 'S1_SAR_SLC',
            'orbitDirection': 'ascending',
        }

        module = DataAcquisitionGateway()
        results = module.search(
            geom=config['project']['aoi']['geom'],
            **search_filter,
        )
        assert len(results) > 0

        output_directory = config['project']['data_dir']  # TODO -> GaiaBase
        data_path = None

        try:
            data_path = module.download(
                results[0], quicklook=False, output_dir=output_directory
            )
            assert Path(data_path).exists()
        finally:
            if data_path and Path(data_path).exists():
                Path(data_path).unlink()

    def test_000a_search_bursts(self, pipeline, config):
        aoi = loads(config['project']['aoi']['geom'])
        pipeline._search_bursts(aoi, '2022-01-01', '2022-01-31', 'A')
        assert pipeline.bursts is not None
        assert len(pipeline.bursts) > 0

    def test_000b_download_bursts(self, pipeline, config):
        if pipeline.bursts is None:
            aoi = loads(config['project']['aoi']['geom'])
            pipeline._search_bursts(aoi, '2022-01-01', '2022-01-31', 'A')
            assert pipeline.bursts is not None
            assert len(pipeline.bursts) > 0

        data_dir = config['project']['data_dir']
        pipeline._download_bursts('username', 'password', data_dir)
        assert any(
            item.is_dir() and 'IW' in item.name for item in Path(data_dir).iterdir()
        )

    def test_001_download_orbits(self, pipeline, config):
        """Test downloading orbit files for Sentinel-1 SLC data."""
        data_dir = config['project']['data_dir']
        pipeline._download_orbits(data_dir)
        assert pipeline.s1 is not None
        assert not pipeline.s1.empty
        eof_files = [f for f in os.listdir(data_dir) if f.endswith('.EOF')]
        assert len(eof_files) > 0

    def test_002_download_dem(self, pipeline, config):
        """Test downloading DEM for Sentinel-1 SLC data."""
        data_dir = Path(config['project']['data_dir'])
        dem_path = data_dir / 'dem.nc'
        aoi = self._get_bbox(config)
        pipeline._download_dem(aoi, dem_path)
        assert pipeline.dem.exists()
        assert os.path.isfile(pipeline.dem)
        with xr.open_dataset(pipeline.dem) as ds:
            assert len(ds.data_vars) > 0
            first_var = list(ds.data_vars)[0]
            assert ds[first_var].size > 0

    def test_003_download_landmask(self, pipeline, config):
        """Test downloading landmask for Sentinel-1 SLC data."""
        data_dir = Path(config['project']['data_dir'])
        landmask_path = data_dir / 'landmask.nc'
        aoi = self._get_bbox(config)
        pipeline._download_landmask(aoi, landmask_path)
        assert pipeline.landmask.exists()
        assert os.path.isfile(pipeline.landmask)
        with xr.open_dataset(pipeline.landmask) as ds:
            assert len(ds.data_vars) > 0
            first_var = list(ds.data_vars)[0]
            assert ds[first_var].size > 0

    def test_004_run_dask_cluster(self, pipeline):
        """Test running local dusk cluster for computation."""
        dask_kwargs = {
            'silence_logs': 'CRITICAL',
            'n_workers': 2,
            'threads_per_worker': 2,
            'memory_limit': '6GB'
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            assert pipeline.client is not None
            assert isinstance(pipeline.client, Client)
            assert pipeline.client.status == "running"

            worker_info = pipeline.client.scheduler_info()['workers']
            assert len(worker_info) == 2

            def square(x):
                return x ** 2

            future = pipeline.client.submit(square, 10)
            result = future.result()
            assert result == 100

        assert pipeline.client is None or pipeline.client.status == "closed"

    def test_005_stack_scenes(self, pipeline, config):
        """Test stacking Sentinel-1 SLC data."""
        data_dir = Path(config['project']['data_dir'])
        work_dir = data_dir / 'workdir'
        pipeline._stack_scenes(data_dir, work_dir)
        assert pipeline.s1 is not None
        assert not pipeline.s1.empty
        assert pipeline.sbas is not None
        df_stack = pipeline.sbas.to_dataframe()
        assert len(df_stack) == len(pipeline.s1)
        assert len(df_stack) > 1

    def test_006_reframe_scenes(self, pipeline, config):
        """Test reframing Sentinel-1 SLC data."""
        data_dir = Path(config['project']['data_dir'])
        work_dir = data_dir / 'workdir'
        if pipeline.sbas is None:
            pipeline._stack_scenes(data_dir, work_dir)
        aoi = loads(config['project']['aoi']['geom'])
        pipeline._reframe_scenes(aoi)
        tiff_files = [f for f in os.listdir(work_dir) if f.endswith('.tiff')]
        assert len(tiff_files) > 0

    def test_007_load_dem_and_landmask(self, pipeline, config):
        """Test loading DEM and landmask for Sentinel-1 SLC data."""
        aoi = loads(config['project']['aoi']['geom'])
        data_dir = Path(config['project']['data_dir'])
        work_dir = data_dir / 'workdir'
        if pipeline.dem is None:
            dem_path = data_dir / 'dem.nc'
            pipeline._download_dem(aoi, dem_path)
        if pipeline.landmask is None:
            landmask_path = data_dir / 'landmask.nc'
            pipeline._download_landmask(aoi, landmask_path)
        if pipeline.sbas is None:
            pipeline._stack_scenes(data_dir, work_dir)
            pipeline._reframe_scenes(aoi)

        dask_kwargs = {
            'silence_logs': 'CRITICAL',
            'n_workers': 2,
            'threads_per_worker': 2,
            'memory_limit': '6GB'
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            assert (work_dir / "landmask.nc").exists()
            assert (work_dir / "DEM_WGS84.nc").exists()

    def test_run_workflow(self, pipeline, config):
        pass
