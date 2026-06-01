from pathlib import Path
import os
from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry
import pytest
import xarray as xr
from dask.distributed import Client
import numpy as np
import pandas as pd
import json

from lib.config import ProjectConfigReader, SettingsReader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from tests.utils import TestUtils


@pytest.fixture(scope='class')
def pipeline():
    module = PreprocessingPipelines()
    return module.pipelines['sentinel1']


@pytest.fixture(scope='class')
def config():
    return ProjectConfigReader(
        TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    )


@dataclass
class ProjectContext:
    """A container for shared project variables."""

    aoi: BaseGeometry
    data_dir: Path
    dem_path: Path
    landmask_path: Path
    work_dir: Path
    result_dir: Path


@pytest.fixture(scope='class')
def ctx(config, glob_config):
    """Bundles multiple config values into one object."""
    base_dir = Path(glob_config['storage']['data_dir']).resolve()
    data_dir = base_dir / 'sentinel1'
    return ProjectContext(
        aoi=config.aoi(),
        data_dir=data_dir,
        dem_path=data_dir / 'dem.nc',
        landmask_path=data_dir / 'landmask.nc',
        work_dir=data_dir / 'workdir',
        result_dir=data_dir / 'results',
    )


@pytest.fixture(scope='class')
def glob_config():
    return SettingsReader()


@pytest.fixture(scope='class', autouse=True)
def dask_cluster(pipeline, glob_config):
    """Starts a Dask cluster for the duration of the test class."""
    dask_kwargs = {
        'silence_logs': glob_config['dask_parameters']['silence_logs'],
        'n_workers': glob_config['dask_parameters']['n_workers'],
        'threads_per_worker': glob_config['dask_parameters']['threads_per_worker'],
        'memory_limit': glob_config['dask_parameters']['memory_limit'],
    }

    pipeline._run_dask_cluster(**dask_kwargs)

    yield pipeline.client

    if pipeline.client:
        pipeline.client.close()


class TestSentinel1Workflow:
    def test_config(self, config, glob_config):
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
            geom=ctx.aoi,
            start=search_filter['start'],
            end=search_filter['end'],
            direction=search_filter['orbitDirection'],
        )
        assert results is not None
        assert len(results) > 0

        datadir = module.backend.download(results, target_dir=ctx.data_dir)
        safe_products = list(Path(datadir).glob('*.SAFE/'))
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
        assert len(worker_info) == SettingsReader()['dask_parameters']['n_workers']

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
        assert pipeline.vel_ll is not None
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

    def test_016_environmental_database(self, pipeline, ctx):
        """Test creating environmental database."""
        pipeline._environmental_database(ctx.aoi, ctx.result_dir)
        climate_path = ctx.result_dir / 'climate_daily_db.csv'
        air_path = ctx.result_dir / 'air_quality_daily_db.csv'
        manifest_path = ctx.result_dir / 'envdb_manifest.json'

        assert climate_path.is_file()
        assert air_path.is_file()
        assert manifest_path.is_file()

        df_climate = pd.read_csv(climate_path)
        assert not df_climate.empty
        assert 'precip_7d_mm' in df_climate.columns

        df_air = pd.read_csv(air_path)
        assert not df_air.empty
        assert 'pm10' in df_air.columns

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        assert 'detected_timezone' in manifest
        assert 'locations' in manifest
        assert len(manifest['locations']) >= 1

    def test_017_compute_risk_database(self, pipeline, ctx):
        """Test creating risk database based on environmental data and displacements."""
        pipeline._compute_risk_database(ctx.result_dir)
        csv_out = ctx.result_dir / 'final_risk_database.csv'
        gov_path = ctx.result_dir / 'risk_governance.json'

        assert csv_out.is_file()
        assert gov_path.is_file()

        df_result = pd.read_csv(csv_out)
        assert not df_result.empty

        expected_cols = [
            'risk_score_0to100',
            'risk_class',
            'UTM_E',
            'UTM_N',
            'cell_id',
            'climate_precip_7d_mm',
        ]
        for col in expected_cols:
            assert col in df_result.columns

        with open(gov_path, 'r') as f:
            gov = json.load(f)

        assert 'thresholds' in gov
        assert 'weights' in gov
        assert 'utm_epsg' in gov
        assert 'calculated_at' in gov

        assert 32600 <= gov['utm_epsg'] <= 32760

    def test_018_export_displacements(self, pipeline, ctx):
        """Test exporting displacements to tif files."""
        pipeline._export_displacements(ctx.result_dir)
        tiff_dir = ctx.result_dir / 'displacements'
        velocity = ctx.result_dir / 'velocity' / 'velocity.tif'
        assert tiff_dir.is_dir()
        tif_files = [f for f in os.listdir(tiff_dir) if f.endswith('.tif')]
        assert len(tif_files) > 0
        assert velocity.is_file()

    def test_019_cleanup(self, pipeline, ctx):
        """Test cleaning work directory."""
        pipeline._cleanup(ctx.work_dir)
        assert not ctx.work_dir.is_dir()
