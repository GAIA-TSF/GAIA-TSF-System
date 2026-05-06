from pathlib import Path
import os
from dataclasses import dataclass

from shapely.wkt import loads
from shapely.geometry.base import BaseGeometry
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


@dataclass
class ProjectContext:
    """A container for shared project variables."""

    aoi: BaseGeometry
    data_dir: Path
    dem_path: Path
    landmask_path: Path
    work_dir: Path


@pytest.fixture(scope='class')
def ctx(config):
    """Bundles multiple config values into one object."""
    project = config['project']
    data_dir = Path(project['data_dir']).resolve()
    return ProjectContext(
        aoi=loads(config['project']['aoi']['geom']),
        data_dir=data_dir,
        dem_path=data_dir / 'dem.nc',
        landmask_path=data_dir / 'landmask.nc',
        work_dir=data_dir / 'workdir',
    )


@pytest.fixture(scope='class', autouse=True)
def dask_cluster(pipeline):
    """Starts a Dask cluster for the duration of the test class."""
    dask_kwargs = {
        'silence_logs': 'CRITICAL',
        'n_workers': 2,
        'threads_per_worker': 2,
        'memory_limit': '6GB',
    }

    pipeline._run_dask_cluster(**dask_kwargs)

    yield pipeline.client

    if pipeline.client:
        pipeline.client.close()


class TestSentinel1Workflow:
    def test_config(self, config):
        """Test project configuration."""
        assert config.is_valid() is True

    def test_000_download_sentinel1_bursts(self, ctx):
        """Test EOU Data Acquisition Gateway to download Sentinel-1 BURST data."""
        search_filter = {
            'start': '2022-01-01',
            'end': '2022-01-31',
            'orbitDirection': 'A',
        }

        module = DataAcquisitionGateway(backend='asf')
        results = module.backend.search(
            aoi=ctx.aoi,
            start=search_filter['start'],
            end=search_filter['end'],
            direction=search_filter['orbitDirection'],
        )
        assert results is not None
        assert len(results) > 0

        datadir = module.backend.download(results, target_dir=ctx.data_dir)
        safe_products = list(datadir.glob('*.SAFE/'))
        assert len(safe_products) > 0

    def test_001_download_orbits(self, pipeline, ctx):
        """Test downloading orbit files for Sentinel-1 SLC data."""
        pipeline._download_orbits(ctx.data_dir)
        eof_files = [f for f in os.listdir(ctx.data_dir) if f.endswith('.EOF')]
        assert len(eof_files) > 0

    def test_002_download_dem(self, pipeline, ctx):
        """Test downloading DEM for Sentinel-1 SLC data."""
        pipeline._download_dem(ctx.aoi, ctx.dem_path)
        assert ctx.dem_path.exists()
        assert ctx.dem_path.is_file()
        with xr.open_dataset(ctx.dem_path) as ds:
            assert len(ds.data_vars) > 0
            first_var = list(ds.data_vars)[0]
            assert ds[first_var].size > 0

    def test_003_download_landmask(self, pipeline, ctx):
        """Test downloading landmask for Sentinel-1 SLC data."""
        pipeline._download_landmask(ctx.aoi, ctx.landmask_path)
        assert ctx.landmask_path.exists()
        assert ctx.landmask_path.is_file()
        with xr.open_dataset(ctx.landmask_path) as ds:
            assert len(ds.data_vars) > 0
            first_var = list(ds.data_vars)[0]
            assert ds[first_var].size > 0

    def test_004_run_dask_cluster(self, pipeline):
        """Test running local dusk cluster for computation."""
        assert pipeline.client is not None
        assert isinstance(pipeline.client, Client)
        assert pipeline.client.status == 'running'

        worker_info = pipeline.client.scheduler_info()['workers']
        assert len(worker_info) == 2

        def square(x):
            return x**2

        future = pipeline.client.submit(square, 10)
        assert future.result() == 100

    def test_005_stack_scenes(self, pipeline, ctx):
        """Test stacking Sentinel-1 SLC data."""
        pipeline._stack_scenes(ctx.data_dir, ctx.work_dir)
        assert pipeline.sbas is not None
        df_stack = pipeline.sbas.to_dataframe()
        assert len(df_stack) > 1
        assert 'datetime' in df_stack.columns
        assert df_stack['datetime'].isnull().sum() == 0
        assert df_stack['datetime'].is_monotonic_increasing

    def test_006_reframe_scenes(self, pipeline, ctx):
        """Test reframing Sentinel-1 SLC data."""
        pipeline._reframe_scenes(ctx.aoi)
        tiff_files = [f for f in os.listdir(ctx.work_dir) if f.endswith('.tiff')]
        assert len(tiff_files) > 0

    def test_007_load_dem_and_landmask(self, pipeline, ctx):
        """Test loading DEM and landmask for Sentinel-1 SLC data."""
        pipeline._load_dem_and_landmask(ctx.aoi, ctx.dem_path, ctx.landmask_path)
        assert (ctx.work_dir / 'landmask.nc').is_file()
        assert (ctx.work_dir / 'DEM_WGS84.nc').is_file()

    def test_008_align_images(self, pipeline, ctx):
        """Test aligning Sentinel-1 SLC data."""
        pipeline._align_images()
        led_files = [f for f in os.listdir(ctx.work_dir) if f.endswith('.LED')]
        assert len(led_files) > 0
        prm_files = [f for f in os.listdir(ctx.work_dir) if f.endswith('.PRM')]
        assert len(prm_files) > 0
        slc_files = [f for f in os.listdir(ctx.work_dir) if f.endswith('.SLC')]
        assert len(slc_files) > 0

    def test_009_geocoding_transform(self, pipeline, ctx):
        """Test geocoding Sentinel-1 SLC data."""
        pipeline._geocoding_transform()
        grd_files = [f for f in os.listdir(ctx.work_dir) if f.endswith('.grd')]
        assert len(grd_files) > 0

    def test_010_find_optimal_network(self, pipeline):
        """Test finding optimal SBAS network."""
        pipeline._find_optimal_network()
        assert pipeline.baseline_pairs is not None
        assert len(pipeline.baseline_pairs) > 0
        required_cols = {'ref', 'rep', 'pair'}
        assert required_cols.issubset(pipeline.baseline_pairs.columns)

    def test_011_compute_interferograms(self, pipeline):
        """Test computing interferograms."""
        pipeline._compute_interferograms()
        assert pipeline.corr is not None
        assert pipeline.intf is not None
        assert len(pipeline.intf.pair) == len(pipeline.baseline_pairs)

    def test_012_unwrap_phase(self, pipeline):
        """Test unwrapping phase."""
        pipeline._unwrap_phase()
        assert pipeline.unwrap is not None
        unwrapped_phase = pipeline.unwrap.phase
        expected_pairs = len(pipeline.baseline_pairs)
        actual_pairs = len(unwrapped_phase.pair)
        assert actual_pairs == expected_pairs
        sample_data = unwrapped_phase.isel(pair=0).values
        assert np.any(~np.isnan(sample_data))
        assert np.nanstd(sample_data) > 0

    def test_013_detrend_phase(self, pipeline):
        """Test detrending phases."""
        pipeline._detrend_phase()
        assert pipeline.detrend is not None
        assert 'pair' in pipeline.detrend.dims
        val = pipeline.detrend.isel(pair=0).mean().compute()
        assert not np.isnan(val)

    def test_014_compute_displacement(self, pipeline):
        """Test computing displacements."""
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

    def test_015_compute_risk(self, pipeline):
        """Test computing risk map."""
        pipeline._compute_risk()
        assert pipeline.risk_map is not None
        assert not pipeline.risk_map.isnull().all()
        assert pipeline.failure_flag is not None
        assert pipeline.risk_map.dims == pipeline.disp_ll.isel(date=0, drop=True).dims
        assert np.allclose(pipeline.risk_map.lat.values, pipeline.disp_ll.lat.values)
        assert np.allclose(pipeline.risk_map.lon.values, pipeline.disp_ll.lon.values)
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
