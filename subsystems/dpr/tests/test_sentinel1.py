from pathlib import Path
import os
import re

from shapely.wkt import loads
from shapely.geometry import Polygon
import numpy as np
import pytest

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

    def test_001_search_bursts(self, pipeline, config):
        """Test Sentinel-1 BURST data search using ASF."""
        aoi = loads(config['project']['aoi']['geom'])
        pipeline._search_bursts(aoi, '2022-07-01', '2022-10-30', 'A')
        assert pipeline.bursts is not None
        assert len(pipeline.bursts) > 0

    def test_002_download_bursts(self, pipeline, config):
        """Test Sentinel-1 BURST data download using ASF."""
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
        """Test downloading orbit files for Sentinel-1 BURST data."""
        data_dir = config['project']['data_dir']
        pipeline._download_orbits(data_dir)
        assert pipeline.s1 is not None
        assert not pipeline.s1.df.empty
        eof_files = [f for f in os.listdir(data_dir) if f.endswith('.EOF')]
        assert len(eof_files) > 0

    def test_004_download_dem_baseline(self, pipeline, config):
        """Test downloading DEM baseline."""
        bbox = self._get_bbox(config)
        pipeline._download_dem_baseline(bbox)
        assert pipeline.dem_da is not None
        assert pipeline.dem_da.rio.crs.to_epsg() == 4326
        assert 'lat' in pipeline.dem_da.coords
        assert 'lon' in pipeline.dem_da.coords
        assert pipeline.dem_da.ndim == 2

    def test_005_lidar_infill(self, pipeline, config):
        """Test infilling DEM with LiDAR data."""
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
        """Test saving DEM to disk."""
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))
            assert pipeline.dem_da is not None

        base_dir = Path(config['project']['data_dir'])
        output_dem = base_dir / 'dem.nc'
        pipeline._save_composite_dem(output_dem)
        assert output_dem.exists()

    def test_007_clip_dem(self, pipeline, config):
        """Test clipping DEM."""
        if pipeline.dem_da is None:
            pipeline._download_dem_baseline(self._get_bbox(config))
            assert pipeline.dem_da is not None

        aoi = loads(config['project']['aoi']['geom'])
        pipeline._clip_dem(aoi)
        assert pipeline.dem_masked is not None
        assert pipeline.dem_cropped is not None
        assert not np.isnan(pipeline.dem_masked.values).all()
        assert not np.isnan(pipeline.dem_cropped.values).all()

    def test_008_save_landmask(self, pipeline, config):
        """Test saving landmask based on DEM to disk."""
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
        """Test linking Sentinel-1 BURST data with DEM."""
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
        """Test finding reference date from Sentinel-1 BURST data."""
        if pipeline.bursts is None:
            aoi = loads(config['project']['aoi']['geom'])
            pipeline._search_bursts(aoi, '2022-07-01', '2022-10-30', 'A')
            assert pipeline.bursts is not None
            assert len(pipeline.bursts) > 0

        pipeline._infer_ref_date()
        assert pipeline.ref_date is not None
        assert isinstance(pipeline.ref_date, str)
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', pipeline.ref_date)

    def test_011_transform_to_zarr(self, pipeline, config):
        """Test transforming Sentinel-1 BURST data to georeferenced zarr format."""
        base_dir = Path(config['project']['data_dir'])
        dem_file = base_dir / 'dem.nc'

        if dem_file.exists():
            zarr_dir = base_dir / 'zarrdir'
            zarr_dir.mkdir(exist_ok=True)
            assert zarr_dir.is_dir()

            if pipeline.ref_date is None:
                if pipeline.bursts is None:
                    aoi = loads(config['project']['aoi']['geom'])
                    pipeline._search_bursts(aoi, '2022-07-01', '2022-10-30', 'A')
                pipeline._infer_ref_date()

            pipeline._transform_to_zarr(dem_file, base_dir, zarr_dir)
            assert any(zarr_dir.iterdir())

    def test_012_get_geometries(self, pipeline, config):
        """Test getting AOI and centroid geometries in UTM coordinates."""
        aoi = loads(config['project']['aoi']['geom'])
        pipeline._get_geometries(aoi, 'EPSG:32735')
        assert pipeline.centroid_utm is not None
        assert pipeline.aoi_utm is not None
        assert pipeline.centroid_utm_off is not None

    def test_013_stack_bursts(self, pipeline, config):
        """Test loading georeferenced BURST data into stack."""
        base_dir = Path(config['project']['data_dir'])
        zarr_dir = base_dir / 'zarrdir'
        with pipeline:
            pipeline._stack_bursts(str(zarr_dir))
            assert pipeline.stack is not None
            assert len(pipeline.stack) > 0

    def test_014_crop_bursts(self, pipeline, config):
        """Test cropping stacked BURST data by AOI."""
        with pipeline:
            if pipeline.aoi_utm is None:
                aoi = loads(config['project']['aoi']['geom'])
                pipeline._get_geometries(aoi, 'EPSG:32735')
            if pipeline.stack is None:
                base_dir = Path(config['project']['data_dir'])
                zarr_dir = base_dir / 'zarrdir'
                pipeline._stack_bursts(str(zarr_dir))

            pipeline._crop_bursts()
            assert pipeline.stack is not None
            assert len(pipeline.stack) > 0
            # TODO: How to check if it was actually cropped?

    def test_015_compute_baseline(self, pipeline, config):
        """Test computing temporal/perpendicular baseline from BURST data."""
        with pipeline:
            if pipeline.stack is None:
                if pipeline.aoi_utm is None:
                    aoi = loads(config['project']['aoi']['geom'])
                    pipeline._get_geometries(aoi, 'EPSG:32735')
                base_dir = Path(config['project']['data_dir'])
                zarr_dir = base_dir / 'zarrdir'
                pipeline._stack_bursts(str(zarr_dir))
                pipeline._crop_bursts()

            pipeline._compute_baseline(24)
            assert pipeline.baseline is not None
            assert len(pipeline.baseline) > 0

    def test_016_compute_interferogram(self, pipeline, config):
        with pipeline:
            if pipeline.stack is None:
                if pipeline.aoi_utm is None:
                    aoi = loads(config['project']['aoi']['geom'])
                    pipeline._get_geometries(aoi, 'EPSG:32735')
                base_dir = Path(config['project']['data_dir'])
                zarr_dir = base_dir / 'zarrdir'
                pipeline._stack_bursts(str(zarr_dir))
                pipeline._crop_bursts()
                pipeline._compute_baseline(24)

            pipeline._compute_interferogram()
            assert pipeline.mintf is not None
            assert pipeline.mcorr is not None

    def test_017_unwrap_interferogram(self, pipeline, config):
        with pipeline:
            base_dir = Path(config['project']['data_dir'])
            dem_file = base_dir / 'dem.nc'
            if pipeline.stack is None:
                if pipeline.aoi_utm is None:
                    aoi = loads(config['project']['aoi']['geom'])
                    pipeline._get_geometries(aoi, 'EPSG:32735')
                zarr_dir = base_dir / 'zarrdir'
                pipeline._stack_bursts(str(zarr_dir))
                pipeline._crop_bursts()
                pipeline._compute_baseline(24)
                pipeline._compute_interferogram()

            pipeline._unwrap_interferogram(dem_file)
            assert pipeline.mphase is not None

    def test_018_detrend_unwrapped_phase(self, pipeline, config):
        with pipeline:
            base_dir = Path(config['project']['data_dir'])
            dem_file = base_dir / 'dem.nc'
            if pipeline.stack is None:
                if pipeline.aoi_utm is None:
                    aoi = loads(config['project']['aoi']['geom'])
                    pipeline._get_geometries(aoi, 'EPSG:32735')
                zarr_dir = base_dir / 'zarrdir'
                pipeline._stack_bursts(str(zarr_dir))
                pipeline._crop_bursts()
                pipeline._compute_baseline(24)
                pipeline._compute_interferogram()
                pipeline._unwrap_interferogram(dem_file)

            pipeline._detrend_unwrapped_phase()
            assert pipeline.mphase_detrend is not None

    def test_019_compute_displacement(self, pipeline, config):
        with pipeline:
            base_dir = Path(config['project']['data_dir'])
            dem_file = base_dir / 'dem.nc'
            if pipeline.stack is None:
                if pipeline.aoi_utm is None:
                    aoi = loads(config['project']['aoi']['geom'])
                    pipeline._get_geometries(aoi, 'EPSG:32735')
                zarr_dir = base_dir / 'zarrdir'
                pipeline._stack_bursts(str(zarr_dir))
                pipeline._crop_bursts()
                pipeline._compute_baseline(24)
                pipeline._compute_interferogram()
                pipeline._unwrap_interferogram(dem_file)
                pipeline._detrend_unwrapped_phase()

            pipeline._compute_displacement()
            assert pipeline.mdisplacement_los is not None
            assert pipeline.mvelocity is not None

    def test_run_workflow(self, pipeline, config):
        pass
