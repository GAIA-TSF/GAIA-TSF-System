from pathlib import Path
import os

from shapely.wkt import loads
from shapely.geometry import Polygon
import pytest
import xarray as xr
from dask.distributed import Client
import numpy as np

from lib.config import ProjectConfigReader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines


@pytest.fixture(scope='class')
def pipeline():
    module = PreprocessingPipelines()
    return module.pipelines['sentinel1']


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

    def test_download_sentinel1_bursts(self, config):
        """Test EOU Data Acquisition Gateway to download Sentinel-1 BURST data."""
        search_filter = {
            'start': '2022-01-01',
            'end': '2022-01-31',
            'orbitDirection': 'A',
        }

        module = DataAcquisitionGateway(backend='asf')
        aoi = loads(config['project']['aoi']['geom'])
        results = module.backend.search(
            aoi=aoi,
            start=search_filter['start'],
            end=search_filter['end'],
            direction=search_filter['orbitDirection'],
        )
        assert results is not None
        assert len(results) > 0

        target_dir = Path(config['project']['data_dir']).resolve()
        datadir = module.backend.download(results, target_dir=target_dir)
        assert any(datadir.iterdir())

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
        aoi = loads(config['project']['aoi']['geom'])
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
        aoi = loads(config['project']['aoi']['geom'])
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            assert pipeline.client is not None
            assert isinstance(pipeline.client, Client)
            assert pipeline.client.status == 'running'

            worker_info = pipeline.client.scheduler_info()['workers']
            assert len(worker_info) == 2

            def square(x):
                return x**2

            future = pipeline.client.submit(square, 10)
            result = future.result()
            assert result == 100

        assert pipeline.client is None or pipeline.client.status == 'closed'

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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            assert (work_dir / 'landmask.nc').exists()
            assert (work_dir / 'DEM_WGS84.nc').exists()
            assert pipeline.dem_masked is not None

    def test_008_align_images(self, pipeline, config):
        """Test aligning Sentinel-1 SLC data."""
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            led_files = [f for f in os.listdir(work_dir) if f.endswith('.LED')]
            assert len(led_files) > 0
            prm_files = [f for f in os.listdir(work_dir) if f.endswith('.PRM')]
            assert len(prm_files) > 0
            slc_files = [f for f in os.listdir(work_dir) if f.endswith('.SLC')]
            assert len(slc_files) > 0

    def test_009_geocoding_transform(self, pipeline, config):
        """Test geocoding Sentinel-1 SLC data."""
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            pipeline._geocoding_transform()
            grd_files = [f for f in os.listdir(work_dir) if f.endswith('.grd')]
            assert len(grd_files) > 0

    def test_010_find_optimal_network(self, pipeline, config):
        """Test finding optimal SBAS network."""
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            pipeline._geocoding_transform()
            pipeline._find_optimal_network()
            assert pipeline.baseline_pairs is not None
            assert len(pipeline.baseline_pairs) > 0
            required_cols = {'ref', 'rep', 'pair'}
            assert required_cols.issubset(pipeline.baseline_pairs.columns)

    def test_011_compute_interferograms(self, pipeline, config):
        """Test computing interferograms."""
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            pipeline._geocoding_transform()
            pipeline._find_optimal_network()
            pipeline._compute_interferograms()
            assert pipeline.corr is not None
            assert pipeline.intf is not None
            assert len(pipeline.intf.pair) == len(pipeline.baseline_pairs)

    def test_012_unwrap_interferograms(self, pipeline, config):
        """Test unwrapping interferograms."""
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            pipeline._geocoding_transform()
            pipeline._find_optimal_network()
            pipeline._compute_interferograms()
            pipeline._unwrap_interferograms()
            assert pipeline.unwrap is not None
            unwrapped_phase = pipeline.unwrap.phase
            expected_pairs = len(pipeline.baseline_pairs)
            actual_pairs = len(unwrapped_phase.pair)
            assert actual_pairs == expected_pairs
            sample_data = unwrapped_phase.isel(pair=0).values
            assert np.any(~np.isnan(sample_data))
            assert np.nanstd(sample_data) > 0

    def test_013_detrend_phase(self, pipeline, config):
        """Test detrending phases."""
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            pipeline._geocoding_transform()
            pipeline._find_optimal_network()
            pipeline._compute_interferograms()
            pipeline._unwrap_interferograms()
            pipeline._detrend_phase()
            assert pipeline.detrend is not None
            assert 'pair' in pipeline.detrend.dims
            val = pipeline.detrend.isel(pair=0).mean().compute()
            assert not np.isnan(val)

    def test_014_compute_displacement(self, pipeline, config):
        """Test computing displacements."""
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
            'memory_limit': '6GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            pipeline._geocoding_transform()
            pipeline._find_optimal_network()
            pipeline._compute_interferograms()
            pipeline._unwrap_interferograms()
            pipeline._detrend_phase()
            pipeline._compute_displacement()
            assert pipeline.disp_ll is not None
            assert pipeline.rmse is not None
            assert 'lat' in pipeline.disp_ll.coords
            assert 'lon' in pipeline.disp_ll.coords
            assert not pipeline.disp_ll.isnull().all()
            assert pipeline.rmse.min() >= 0
            t_dim = next(
                (d for d in pipeline.disp_ll.dims if d in ('date', 'time', 'epoch')),
                None,
            )
            assert t_dim is not None
            assert len(pipeline.disp_ll[t_dim]) > 1
            first_step = pipeline.disp_ll.isel({t_dim: 0})
            assert (first_step == 0).any() or first_step.isnull().all()

    def test_015_compute_risk(self, pipeline, config):
        """Test computing risk map."""
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
            'memory_limit': '8GB',
        }
        with pipeline:
            pipeline._run_dask_cluster(**dask_kwargs)
            pipeline._load_dem_and_landmask(aoi)
            pipeline._align_images()
            pipeline._geocoding_transform()
            pipeline._find_optimal_network()
            pipeline._compute_interferograms()
            pipeline._unwrap_interferograms()
            pipeline._detrend_phase()
            pipeline._compute_displacement()
            pipeline._compute_risk()
            assert pipeline.risk_map is not None
            assert not pipeline.risk_map.isnull().all()
            assert pipeline.failure_flag is not None
            assert (
                pipeline.risk_map.dims == pipeline.disp_ll.isel(date=0, drop=True).dims
            )
            assert np.allclose(
                pipeline.risk_map.lat.values, pipeline.disp_ll.lat.values
            )
            assert np.allclose(
                pipeline.risk_map.lon.values, pipeline.disp_ll.lon.values
            )
            risk_min = pipeline.risk_map.min().compute()
            risk_max = pipeline.risk_map.max().compute()
            assert risk_min >= 0
            assert risk_max <= 9
            unique_flags = np.unique(pipeline.failure_flag.compute())
            assert all(flag in [0, 1] for flag in unique_flags)
            high_risk_areas = pipeline.risk_map.where(pipeline.failure_flag == 1)
            if not high_risk_areas.isnull().all():
                assert high_risk_areas.min(skipna=True) >= 6

    def test_run_workflow(self, pipeline, config):
        # pipeline.configure('WKT...', ...)
        # pipeline.run()
        pass
