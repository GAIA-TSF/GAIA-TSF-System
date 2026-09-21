"""Small numerical regressions: no downloads, GMTSAR executables or cluster."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from pygmtsar import Stack
from shapely.geometry import box, MultiPolygon

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
    return xr.DataArray(values, dims=('pair', 'y', 'x'),
                        coords={'pair': np.arange(values.shape[0]),
                                'y': np.arange(values.shape[1]),
                                'x': np.arange(values.shape[2])})


def test_preserves_broad_deformation(pipeline):
    y, x = np.mgrid[-1:1:40j, -1:1:40j]
    bowl = 5 * np.exp(-(x*x + y*y))
    phase = phase_grid([bowl, 2*bowl])
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline._detrend_phase()
    xr.testing.assert_allclose(pipeline.detrend.compute(), phase)
    pipeline.sbas.gaussian.assert_not_called()


def test_stable_reference_removes_offset_not_deformation(pipeline):
    phase = phase_grid([[[3, 3, 8], [3, 3, 8]], [[7, 7, 17], [7, 7, 17]]])
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline._config['reference_area_wkt'] = box(-.5, -.5, 1.5, 1.5).wkt
    pipeline.sbas.geocode.side_effect = lambda p: p
    pipeline._detrend_phase()
    np.testing.assert_allclose(pipeline.detrend.isel(x=2), [[5, 5], [10, 10]])
    np.testing.assert_allclose(pipeline.detrend.isel(x=0), 0)


def test_empty_reference_fails(pipeline):
    pipeline.unwrap = phase_grid([[[1, 2], [3, 4]]]).to_dataset(name='phase')
    pipeline._config['reference_area_wkt'] = box(20, 20, 30, 30).wkt
    pipeline.sbas.geocode.side_effect = lambda p: p
    with pytest.raises(ValueError, match='no valid phase'):
        pipeline._detrend_phase()


def test_reference_failure_reports_pair_and_stage_counts(pipeline, tmp_path):
    phase = phase_grid([[[np.nan, np.nan], [np.nan, np.nan]], [[1, 1], [1, 1]]])
    phase = phase.assign_coords(pair=['2020-01-01 2020-01-13', '2020-01-13 2020-01-25'])
    pipeline.unwrap = phase.to_dataset(name='phase')
    pipeline.corr_unwrap = xr.ones_like(phase) * .8
    pipeline._config.update(reference_area_wkt=box(-1, -1, 2, 2).wkt, result_dir=tmp_path)
    pipeline.sbas.geocode.side_effect = lambda p: p
    with pytest.raises(ValueError, match='2020-01-01 2020-01-13'):
        pipeline._detrend_phase()
    report = pd.read_csv(tmp_path / 'quality/reference_phase_diagnostics.csv')
    assert report.reference_mask_pixels.tolist() == [4, 4]
    assert report.reference_coherent_pixels.tolist() == [4, 4]
    assert report.reference_valid_phase_pixels.tolist() == [0, 4]
    assert report.all_valid_phase_pixels.tolist() == [0, 4]


def test_all_reference_polygons_contribute(pipeline):
    pipeline.unwrap = phase_grid([[[2, 20, 6], [2, 20, 6]]]).to_dataset(name='phase')
    reference = MultiPolygon([box(-.4, -.4, .4, 1.4), box(1.6, -.4, 2.4, 1.4)])
    pipeline._config['reference_area_wkt'] = reference.wkt
    pipeline.sbas.geocode.side_effect = lambda polygon: polygon
    pipeline._detrend_phase()
    np.testing.assert_allclose(pipeline.detrend, [[[-2, 16, 2], [-2, 16, 2]]])
    assert pipeline.sbas.geocode.call_count == 2


@pytest.mark.parametrize('site,count', [('cadia', 9), ('jagersfontein', 3)])
def test_configured_site_reference_files(site, count):
    from pathlib import Path
    from shapely.wkt import loads
    from lib.config import ProjectConfigReader
    from subsystems.dpr.run_s1_eou_dpr import load_reference_area
    from tests.utils import TestUtils

    config_path = Path(TestUtils.get_project_config_path('slope_monitoring_' + site))
    config = ProjectConfigReader(config_path)
    reference = loads(load_reference_area(
        config_path.parent / config['sentinel1']['reference_area'], config.aoi()))
    assert isinstance(reference, MultiPolygon)
    assert len(reference.geoms) == count


def test_coherence_mask_is_boolean_and_survives_unwrap(pipeline):
    pipeline.intf = phase_grid([[[1, 2], [3, 1]]])
    pipeline.corr = phase_grid([[[.8, .1], [np.nan, .7]]])
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
    pipeline.intf = phase_grid([[[np.pi-.1, -np.pi+.1]]])
    pipeline.corr = xr.ones_like(pipeline.intf) * .8
    pipeline.sbas.decimator.return_value = lambda a: a.coarsen(x=2).mean()
    pipeline.sbas.unwrap_snaphu.side_effect = lambda phase, weight, conncomp=False: phase.to_dataset(name='phase')
    pipeline._unwrap_phase()
    assert abs(float(pipeline.unwrap.phase.item())) == pytest.approx(np.pi)


def test_unwrap_checkpoints_preserve_inputs_and_component_labels(pipeline, tmp_path):
    from subsystems.dpr.preprocessing_pipelines.insar_checkpoints import PairCheckpoints
    pipeline._checkpoints = PairCheckpoints(tmp_path, {})
    pipeline.intf = phase_grid([[[1, 2], [3, 1]]])
    pipeline.corr = phase_grid([[[.8, .1], [np.nan, .7]]])
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
    data = xr.DataArray(np.array([[[1j, 2j]], [[empty_value, empty_value]]]),
                        dims=('date', 'y', 'x'),
                        coords={'date': pd.to_datetime(['2017-09-22', '2017-10-04'])})
    pipeline._config['result_dir'] = tmp_path
    with pytest.raises(ValueError, match='2017-10-04'):
        pipeline._check_slc_coverage(data)
    report = pd.read_csv(tmp_path / 'quality/slc_coverage.csv')
    assert report.valid_slc_pixels.tolist() == [2, 0]


def test_disconnected_epochs_are_not_observed_zeroes():
    refs, reps = np.array([0, 1, 2]), np.array([1, 2, 3])
    np.testing.assert_array_equal(
        Sentinel1Pipeline._reference_connected(np.array([True, False, True]), refs, reps, 4),
        [True, True, False, False])
    assert not Sentinel1Pipeline._reference_connected(np.zeros(3, bool), refs, reps, 4).any()


@pytest.mark.parametrize('missing_bridge', [False, True])
def test_solver_units_reference_epoch_quality_and_export(pipeline, tmp_path, missing_bridge):
    # Exercise the installed PyGMTSAR least-squares and velocity implementations.
    stack = object.__new__(Stack)
    phase = phase_grid([np.ones((2, 2)), np.ones((2, 2)) * 2])
    dates = pd.to_datetime(['2020-01-01', '2020-01-13', '2020-01-25'])
    phase = phase.assign_coords(ref=('pair', dates[:-1]), rep=('pair', dates[1:])).chunk()
    pipeline.detrend = phase
    pipeline.corr = xr.ones_like(phase) * .8
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
        target = tmp_path / 'los' / source.name.replace('disp_', 'los_', 1)
        assert target.read_bytes() == source.read_bytes()
    with rasterio.open(tmp_path / 'velocity/velocity.tif') as src:
        assert src.tags()['units'] == 'mm/year'
        assert np.isnan(src.nodata)
    assert len(list((tmp_path / 'quality').glob('*.tif'))) == 3


def test_no_coherent_pixels_fails(pipeline):
    pipeline.intf = phase_grid([[[1, 2], [3, 4]]])
    pipeline.corr = xr.ones_like(pipeline.intf) * .1
    pipeline.sbas.decimator.return_value = lambda a: a
    with pytest.raises(RuntimeError, match='No pixels'):
        pipeline._unwrap_phase()
    pipeline.sbas.unwrap_snaphu.assert_not_called()


def test_rmse_in_mm_is_not_wrapped(pipeline):
    dates = pd.to_datetime(['2020-01-01', '2020-01-13', '2020-01-25'])
    phase = phase_grid([np.ones((2, 2))*10, np.ones((2, 2))*10])
    phase = phase.assign_coords(ref=('pair', dates[:-1]), rep=('pair', dates[1:]))
    pipeline.detrend = phase
    pipeline.corr = pipeline.corr_unwrap = xr.ones_like(phase) * .8
    pipeline.sbas.get_pairs.side_effect = object.__new__(Stack).get_pairs
    pipeline.sbas.lstsq.return_value = xr.DataArray(np.zeros((3, 2, 2)),
        dims=('date', 'y', 'x'), coords={'date': dates, 'y': [0, 1], 'x': [0, 1]})
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
