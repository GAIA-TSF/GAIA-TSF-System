"""Small numerical regressions: no downloads, GMTSAR executables or cluster."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from pygmtsar import Stack
from pathlib import PosixPath

from shapely.geometry import box, MultiPolygon, Polygon

from lib.exceptions import GaiaConfigError

from subsystems.dpr.preprocessing_pipelines.sentinel1 import Sentinel1Pipeline


@pytest.fixture
def pipeline():
    p = Sentinel1Pipeline()
    p._configure()
    p._config = {}
    p.sbas = Mock()
    return p


def phase_grid(values):
    values = np.asarray(values, dtype=float)
    return xr.DataArray(
        values,
        dims=('pair', 'y', 'x'),
        coords={
            'pair': np.arange(values.shape[0]),
            'y': np.arange(values.shape[1]),
            'x': np.arange(values.shape[2]),
        },
    )


def test_preserves_broad_deformation(pipeline):
    y, x = np.mgrid[-1:1:40j, -1:1:40j]
    bowl = 5 * np.exp(-(x * x + y * y))
    phase = phase_grid([bowl, 2 * bowl])
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline._detrend_phase()
    xr.testing.assert_allclose(pipeline.detrend.compute(), phase)
    pipeline.sbas.gaussian.assert_not_called()


def test_stable_reference_removes_offset_not_deformation(pipeline):
    phase = phase_grid([[[3, 3, 8], [3, 3, 8]], [[7, 7, 17], [7, 7, 17]]])
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline._reference_geometry = box(-0.5, -0.5, 1.5, 1.5)
    pipeline._config['reference_min_pixels'] = 1
    pipeline.sbas.geocode.side_effect = lambda p: p
    pipeline._detrend_phase()
    np.testing.assert_allclose(pipeline.detrend.isel(x=2), [[5, 5], [10, 10]])
    np.testing.assert_allclose(pipeline.detrend.isel(x=0), 0)


def test_empty_reference_fails(pipeline):
    pipeline.unwrap = phase_grid([[[1, 2], [3, 4]]]).to_dataset(name='phase')
    pipeline._reference_geometry = box(20, 20, 30, 30)
    pipeline.sbas.geocode.side_effect = lambda p: p
    with pytest.raises(ValueError, match='fails the quality checks for all'):
        pipeline._detrend_phase()


def test_pair_without_reference_phase_is_dropped(pipeline, tmp_path):
    phase = phase_grid([[[np.nan, np.nan], [np.nan, np.nan]], [[1, 1], [1, 1]]])
    phase = phase.assign_coords(pair=['2020-01-01 2020-01-13', '2020-01-13 2020-01-25'])
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline.corr_unwrap = xr.ones_like(phase) * 0.8
    pipeline._reference_geometry = box(-1, -1, 2, 2)
    pipeline._config.update(
        result_dir=tmp_path,
        reference_min_pixels=1,
    )
    pipeline.sbas.geocode.side_effect = lambda p: p
    pipeline._detrend_phase()
    report = pd.read_csv(tmp_path / 'quality/reference_phase_diagnostics.csv')
    assert report.reference_mask_pixels.tolist() == [4, 4]
    assert report.reference_coherent_pixels.tolist() == [4, 4]
    assert report.reference_valid_phase_pixels.tolist() == [0, 4]
    assert report.all_valid_phase_pixels.tolist() == [0, 4]
    assert report.reference_ok.tolist() == [False, True]
    assert pipeline.detrend.isel(pair=0).isnull().all()
    np.testing.assert_allclose(pipeline.detrend.isel(pair=1), 0)


def test_all_reference_polygons_contribute(pipeline):
    pipeline.unwrap = phase_grid([[[2, 20, 6], [2, 20, 6]]]).to_dataset(name='phase')
    reference = MultiPolygon([box(-0.4, -0.4, 0.4, 1.4), box(1.6, -0.4, 2.4, 1.4)])
    pipeline._reference_geometry = reference
    pipeline._config['reference_min_pixels'] = 1
    pipeline.sbas.geocode.side_effect = lambda polygon: polygon
    pipeline._detrend_phase()
    np.testing.assert_allclose(pipeline.detrend, [[[-2, 16, 2], [-2, 16, 2]]])
    assert pipeline.sbas.geocode.call_count == 2


def test_reference_offset_is_coherence_weighted_median(pipeline):
    # one unwrapping-error pixel (+2π) in the reference must not shift the offset
    phase = phase_grid([[[1, 1, 1, 1 + 2 * np.pi, 9], [1, 1, 1, 1, 9]]])
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline.corr_unwrap = xr.full_like(phase, 0.8)
    pipeline._reference_geometry = box(-0.5, -0.5, 3.5, 1.5)
    pipeline._config.update(reference_min_pixels=1)
    pipeline.sbas.geocode.side_effect = lambda p: p
    pipeline._detrend_phase()
    np.testing.assert_allclose(pipeline.detrend.isel(x=4), 8)


def test_reference_uses_dominant_component_and_masks_others(pipeline):
    phase = phase_grid([[[1, 1, 1, 50, 7], [1, 1, 1, 50, 7]]])
    comp = xr.DataArray(np.array([[[1, 1, 1, 2, 1], [1, 1, 1, 2, 1]]]), dims=phase.dims, coords=phase.coords)
    pipeline.unwrap = xr.Dataset({'phase': phase, 'conncomp': comp})
    pipeline._reference_geometry = box(-0.5, -0.5, 3.5, 1.5)
    pipeline._config.update(reference_min_pixels=1)
    pipeline.sbas.geocode.side_effect = lambda p: p
    pipeline._detrend_phase()
    np.testing.assert_allclose(pipeline.detrend.isel(x=4), 6)
    assert pipeline.detrend.isel(x=3).isnull().all()


def test_low_coherence_reference_pair_is_dropped(pipeline):
    phase = phase_grid([[[1, 1, 5], [1, 1, 5]], [[2, 2, 6], [2, 2, 6]]])
    corr = xr.full_like(phase, 0.8)
    corr[1] = 0.25
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline.corr_unwrap = corr
    pipeline._reference_geometry = box(-0.5, -0.5, 1.5, 1.5)
    pipeline._config.update(reference_min_pixels=1)
    pipeline.sbas.geocode.side_effect = lambda p: p
    pipeline._detrend_phase()
    np.testing.assert_allclose(pipeline.detrend.isel(pair=0, x=2), 4)
    assert pipeline.detrend.isel(pair=1).isnull().all()


@pytest.mark.parametrize('site,count', [('cadia', 9), ('jagersfontein', 3)])
def test_configured_site_reference_files(pipeline, site, count):
    from pathlib import Path
    from lib.config import ProjectConfigReader
    from tests.utils import TestUtils

    config_path = Path(TestUtils.get_project_config_path('slope_monitoring_' + site))
    config = ProjectConfigReader(config_path)
    reference = pipeline._load_reference_area(
        config_path.parent / config['sentinel1']['reference_area'], config.aoi()
    )
    assert isinstance(reference, MultiPolygon)
    assert len(reference.geoms) == count


AOI = box(-7.2, 37.5, -7.0, 37.7)  # EPSG:4326


def write_reference(path, geometries, crs=4326):
    import geopandas as gpd

    gpd.GeoDataFrame(geometry=geometries, crs=crs).to_file(path)
    return path


def test_reference_file_in_wgs84_is_loaded(pipeline, tmp_path):
    path = write_reference(tmp_path / 'ref.gpkg', [box(-7.12, 37.60, -7.11, 37.61)])
    polygon = pipeline._load_reference_area(path, AOI)
    assert isinstance(polygon, Polygon)
    np.testing.assert_allclose(polygon.bounds, (-7.12, 37.60, -7.11, 37.61))


def test_reference_features_are_merged(pipeline, tmp_path):
    path = write_reference(
        tmp_path / 'ref.gpkg',
        [box(-7.12, 37.60, -7.11, 37.61), box(-7.05, 37.60, -7.04, 37.61)],
    )
    polygon = pipeline._load_reference_area(path, AOI)
    assert isinstance(polygon, MultiPolygon) and len(polygon.geoms) == 2


def test_reference_in_other_crs_fails(pipeline, tmp_path):
    path = write_reference(
        tmp_path / 'ref.gpkg', [box(666100, 4163100, 666400, 4163400)], crs=32629
    )
    with pytest.raises(GaiaConfigError, match='EPSG:4326'):
        pipeline._load_reference_area(path, AOI)


def test_reference_outside_aoi_fails(pipeline, tmp_path):
    path = write_reference(tmp_path / 'ref.gpkg', [box(-6.12, 37.60, -6.11, 37.61)])
    with pytest.raises(GaiaConfigError, match='inside the processing AOI'):
        pipeline._load_reference_area(path, AOI)


def test_reference_without_crs_fails(pipeline, tmp_path):
    path = write_reference(tmp_path / 'ref.gpkg', [box(0, 0, 1, 1)], crs=None)
    with pytest.raises(GaiaConfigError, match='EPSG:4326'):
        pipeline._load_reference_area(path, AOI)


def test_missing_reference_file_fails(pipeline, tmp_path):
    with pytest.raises(GaiaConfigError, match='not found'):
        pipeline._load_reference_area(tmp_path / 'missing.gpkg', AOI)


def test_reference_path_param_is_optional():
    p = Sentinel1Pipeline()
    params = p.metadata['params']['reference_stable_area']
    assert params['dtype'] is PosixPath and params.get('required') is False
    assert 'default' not in params


def test_coherence_mask_is_boolean_and_survives_unwrap(pipeline):
    pipeline.intf = phase_grid([[[1, 2], [3, 1]]])
    pipeline.corr = phase_grid([[[0.8, 0.1], [np.nan, 0.7]]])
    pipeline.sbas.decimator.return_value = lambda a: a

    def unwrap(phase, weight, conncomp=False):
        assert conncomp
        assert np.isnan(phase.values[0, 0, 1])
        assert np.isnan(phase.values[0, 1, 0])
        return phase.fillna(123).to_dataset(name='phase')

    pipeline.sbas.unwrap_snaphu.side_effect = unwrap
    pipeline._unwrap_phase()
    assert pipeline.unwrap.phase.count() == 2
    assert pipeline.corr_unwrap.count() == 2


def test_wrapped_phase_decimation_handles_branch_cut(pipeline):
    pipeline.intf = phase_grid([[[np.pi - 0.1, -np.pi + 0.1]]])
    pipeline.corr = xr.ones_like(pipeline.intf) * 0.8
    pipeline.sbas.decimator.return_value = lambda a: a.coarsen(x=2).mean()
    pipeline.sbas.unwrap_snaphu.side_effect = (
        lambda phase, weight, conncomp=False: phase.to_dataset(name='phase')
    )
    pipeline._unwrap_phase()
    assert abs(float(pipeline.unwrap.phase.item())) == pytest.approx(np.pi)


def test_unwrap_checkpoints_preserve_inputs_and_component_labels(pipeline, tmp_path):
    from subsystems.dpr.preprocessing_pipelines.insar_checkpoints import PairCheckpoints

    pipeline._checkpoints = PairCheckpoints(tmp_path, {})
    pipeline.intf = phase_grid([[[1, 2], [3, 1]]])
    pipeline.corr = phase_grid([[[0.8, 0.1], [np.nan, 0.7]]])
    pipeline.sbas.decimator.return_value = lambda a: a
    pipeline.sbas.snaphu_config.return_value = 'DEFOMAX_CYCLE 0'
    labels = phase_grid([[[1, 0], [0, 2]]])

    def unwrap(phase, weight, conncomp=False):
        assert conncomp
        return xr.Dataset({'phase': phase.fillna(123), 'conncomp': labels})

    pipeline.sbas.unwrap_snaphu.side_effect = unwrap
    pipeline._unwrap_phase()
    with PairCheckpoints.open_stage(pipeline._checkpoints.path, 'wrapped') as saved:
        xr.testing.assert_allclose(saved.coherence, pipeline.corr)
        assert saved.valid_mask.values.tolist() == [[[1, 0], [0, 1]]]
    with PairCheckpoints.open_stage(pipeline._checkpoints.path, 'unwrapped') as saved:
        xr.testing.assert_allclose(saved.unwrapped_phase, pipeline.unwrap.phase)
        xr.testing.assert_allclose(saved.snaphu_component, labels)


def test_coherence_requires_matching_kernels(pipeline):
    with pytest.raises(ValueError, match='same averaging kernel'):
        pipeline._compute_interferograms(intensity_wavelength=20, phase_wavelength=30)


@pytest.mark.parametrize('empty_value', [np.nan, 0j])
def test_empty_slc_is_identified_before_interferograms(pipeline, tmp_path, empty_value):
    data = xr.DataArray(
        np.array([[[1j, 2j]], [[empty_value, empty_value]]]),
        dims=('date', 'y', 'x'),
        coords={'date': pd.to_datetime(['2017-09-22', '2017-10-04'])},
    )
    pipeline._config['result_dir'] = tmp_path
    with pytest.raises(ValueError, match='2017-10-04'):
        pipeline._check_slc_coverage(data)
    report = pd.read_csv(tmp_path / 'quality/slc_coverage.csv')
    assert report.valid_slc_pixels.tolist() == [2, 0]


def test_disconnected_epochs_are_not_observed_zeroes():
    refs, reps = np.array([0, 1, 2]), np.array([1, 2, 3])
    np.testing.assert_array_equal(
        Sentinel1Pipeline._reference_connected(
            np.array([True, False, True]), refs, reps, 4
        ),
        [True, True, False, False],
    )
    assert not Sentinel1Pipeline._reference_connected(
        np.zeros(3, bool), refs, reps, 4
    ).any()


@pytest.mark.parametrize('missing_bridge', [False, True])
def test_solver_units_reference_epoch_quality_and_export(
    pipeline, tmp_path, missing_bridge
):
    # Exercise the installed PyGMTSAR least-squares and velocity implementations.
    stack = object.__new__(Stack)
    phase = phase_grid([np.ones((2, 2)), np.ones((2, 2)) * 2])
    dates = pd.to_datetime(['2020-01-01', '2020-01-13', '2020-01-25'])
    phase = phase.assign_coords(
        ref=('pair', dates[:-1]), rep=('pair', dates[1:])
    ).chunk()
    pipeline.detrend = phase
    pipeline.corr = xr.ones_like(phase) * 0.8
    pipeline.corr_unwrap = pipeline.corr.copy()
    if missing_bridge:
        pipeline.detrend[1, 1, 1] = np.nan
        pipeline.corr_unwrap[1, 1, 1] = np.nan
    pipeline.sbas.lstsq.side_effect = stack.lstsq
    pipeline.sbas.get_pairs.side_effect = stack.get_pairs
    pipeline.sbas.los_displacement_mm.side_effect = lambda a: -4.4 * a
    pipeline.sbas.velocity.side_effect = stack.velocity
    pipeline.sbas.ra2ll.side_effect = lambda a: a.rename({'y': 'lat', 'x': 'lon'})
    pipeline.sbas.cropna.side_effect = lambda a: a
    pipeline._compute_displacement()
    np.testing.assert_allclose(pipeline.disp_ll[:, 0, 0], [0, -4.4, -13.2], atol=1e-5)
    np.testing.assert_allclose(pipeline.vel_ll[0, 0], -13.2 / 24 * 365.25, rtol=1e-5)
    if missing_bridge:
        assert np.isnan(pipeline.disp_ll[2, 1, 1])
        np.testing.assert_allclose(pipeline.disp_ll[:2, 1, 1], [0, -4.4], atol=1e-5)
    np.testing.assert_allclose(pipeline.rmse, 0, atol=1e-5)
    pipeline._export_displacements(tmp_path)
    assert len(list((tmp_path / 'displacements').glob('*.tif'))) == 3
    for source in (tmp_path / 'displacements').glob('disp_*.tif'):
        target = tmp_path / 'los_cumulative' / source.name.replace('disp_', 'los_', 1)
        assert target.read_bytes() == source.read_bytes()
    assert not (tmp_path / 'los').exists()
    with rasterio.open(tmp_path / 'velocity/velocity.tif') as src:
        assert src.tags()['units'] == 'mm/year'
        assert np.isnan(src.nodata)
    assert len(list((tmp_path / 'quality').glob('*.tif'))) == 3
    assert len(list((tmp_path / 'los_cumulative').glob('*.tif'))) == 3
    assert len(list((tmp_path / 'los_incremental').glob('*.tif'))) == 2
    assert not (tmp_path / 'los_incremental/los_20200101.tif').exists()
    with rasterio.open(tmp_path / 'los_incremental/los_20200125.tif') as src:
        np.testing.assert_allclose(src.read(1)[0, 0], -8.8, atol=1e-5)
        assert src.tags()['units'] == 'mm'
        assert src.tags()['interval_start'].startswith('2020-01-13')
        if missing_bridge:
            assert np.isnan(src.read(1)[1, 1])
    with rasterio.open(tmp_path / 'velocity_interval/velocity_20200125.tif') as src:
        np.testing.assert_allclose(src.read(1)[0, 0], -8.8 / 12, atol=1e-5)
        assert src.tags()['units'] == 'mm/day'
    for product in ('coherence', 'valid_coverage', 'pair_residuals'):
        assert len(list((tmp_path / product).glob('*.tif'))) == 2
    with rasterio.open(tmp_path / 'coherence/coherence_20200113_20200125.tif') as src:
        np.testing.assert_allclose(src.read(1)[0, 0], 0.8)
    with rasterio.open(
        tmp_path / 'pair_residuals/pair_residuals_20200113_20200125.tif'
    ) as src:
        np.testing.assert_allclose(src.read(1)[0, 0], 0, atol=1e-5)
    if missing_bridge:
        with rasterio.open(
            tmp_path / 'valid_coverage/valid_coverage_20200113_20200125.tif'
        ) as src:
            assert src.read(1)[1, 1] == 0


def test_no_coherent_pixels_fails(pipeline):
    pipeline.intf = phase_grid([[[1, 2], [3, 4]]])
    pipeline.corr = xr.ones_like(pipeline.intf) * 0.1
    pipeline.sbas.decimator.return_value = lambda a: a
    with pytest.raises(RuntimeError, match='No pixels'):
        pipeline._unwrap_phase()
    pipeline.sbas.unwrap_snaphu.assert_not_called()


def test_rmse_in_mm_is_not_wrapped(pipeline):
    dates = pd.to_datetime(['2020-01-01', '2020-01-13', '2020-01-25'])
    phase = phase_grid([np.ones((2, 2)) * 10, np.ones((2, 2)) * 10])
    phase = phase.assign_coords(ref=('pair', dates[:-1]), rep=('pair', dates[1:]))
    pipeline.detrend = phase
    pipeline.corr = pipeline.corr_unwrap = xr.ones_like(phase) * 0.8
    pipeline.sbas.get_pairs.side_effect = object.__new__(Stack).get_pairs
    pipeline.sbas.lstsq.return_value = xr.DataArray(
        np.zeros((3, 2, 2)),
        dims=('date', 'y', 'x'),
        coords={'date': dates, 'y': [0, 1], 'x': [0, 1]},
    )
    pipeline.sbas.los_displacement_mm.side_effect = lambda a: a * -4.4
    pipeline.sbas.velocity.side_effect = lambda a: a.isel(date=0, drop=True)
    pipeline.sbas.ra2ll.side_effect = lambda a: a.rename({'y': 'lat', 'x': 'lon'})
    pipeline.sbas.cropna.side_effect = lambda a: a
    pipeline._compute_displacement()
    np.testing.assert_allclose(pipeline.rmse, 44)


def test_coordinate_mismatch_fails_before_inversion(pipeline):
    pipeline.detrend = phase_grid([[[1, 2], [3, 4]]])
    pipeline.corr = xr.ones_like(pipeline.detrend)
    pipeline.corr_unwrap = pipeline.corr.assign_coords(x=[10, 11])
    with pytest.raises(ValueError, match='align'):
        pipeline._compute_displacement()
    pipeline.sbas.lstsq.assert_not_called()
