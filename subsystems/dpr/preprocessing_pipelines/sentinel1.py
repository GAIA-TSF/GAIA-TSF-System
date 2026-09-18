import os
import json
import shutil
import time
from pathlib import Path, PosixPath
from collections import defaultdict
from datetime import datetime, timezone

import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from pygmtsar import S1, Tiles, Stack
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
import pandas as pd
import requests_cache
import openmeteo_requests
from timezonefinder import TimezoneFinder
from retry_requests import retry
from shapely.wkt import loads
from pyproj import Transformer

from .base import PreprocessingBasePipeline
from .insar_diagnostics import InSARDiagnostics
from lib.config import SettingsReader


class Sentinel1Pipeline(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
        'params': {
            'diagnostics_root': {
                'dtype': str,
                'default': '',
                'description': 'Site directory containing diagnostics/<run timestamp>; defaults to result_dir',
            },
            'diagnostics': {
                'dtype': bool,
                'default': True,
                'description': 'Save per-run JSON/CSV statistics, SBAS graph and intermediate PNG diagnostics',
            },
            'reference_area_wkt': {
                'dtype': str,
                'default': '',
                'description': 'Optional stable-ground Polygon/MultiPolygon WKT in EPSG:4326, inside processing AOI',
            },
            'excluded_dates': {
                'dtype': list,
                'default': [],
                'description': 'Acquisition dates to exclude from processing (YYYY-MM-DD)',
            },
            'datadir': {
                'dtype': PosixPath,
                'description': 'Path to the directory with Sentinel-1 SLC BURST data',
            },
            'aoi': {
                'dtype': Polygon,
                'description': 'POLYGON wkt string or Shapely Polygon, coordinates must be in WGS84 (EPSG:4326)',
            },
            'dem_path': {
                'dtype': PosixPath,
                'description': 'Path where the DEM file will be downloaded and later used',
            },
            'landmask_path': {
                'dtype': PosixPath,
                'description': 'Path where the landmask file will be downloaded and later used',
            },
            'workdir': {
                'dtype': PosixPath,
                'description': "Path to the directory where computed data will be stored (cannot be the same as 'datadir')",
            },
            'result_dir': {
                'dtype': PosixPath,
                'description': 'Path to the directory where final results will be stored',
            },
        },
    }

    def _configure(self):
        self._diagnostics = None
        self.client = None
        self.sbas = None
        self.baseline_pairs = None
        self.corr = None
        self.corr_unwrap = None
        self.intf = None
        self.unwrap = None
        self.detrend = None
        self.disp_ll = None
        self.vel_ll = None
        self.rmse = None
        self.quality_ll = {}
        self.risk_map = None
        self.failure_flag = None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def _download_orbits(self, datadir):
        """Download precise orbit files for Sentinel-1 BURST data.

        :param Path datadir: Path to the directory with Sentinel-1 SLC BURST data.
        :return: None.
        """
        if not any(datadir.glob('*.SAFE/')):
            raise FileNotFoundError(f"No '.SAFE' directories found in {datadir}.")
        self.logger.info('Downloading precise orbit files.')
        s1 = S1.scan_slc(datadir)
        S1.download_orbits(datadir, s1)

    def _download_dem(self, aoi, output_dem):
        """Download DEM.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_dem: Path to output DEM file.
        :return: None
        """
        self.logger.info('Downloading DEM for AOI.')
        # inconsistent tile shapes detected with GLO -> switching to SRTM
        Tiles().download_dem(aoi, filename=output_dem, provider='SRTM', skip_exist=True)

    def _download_landmask(self, aoi, output_landmask):
        """Download landmask.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_landmask: Path to output landmask file.
        :return: None
        """
        self.logger.info('Downloading landmask for AOI.')
        Tiles().download_landmask(
            aoi, filename=output_landmask, skip_exist=True
        ).fillna(0)

    def _run_dask_cluster(self, **kwargs):
        """Run Dask Cluster Client for computation.

        :return: None
        """
        if not hasattr(self, 'client'):
            self._configure()
        if self.client is None:
            self.client = Client(**kwargs)

    def _stack_scenes(self, datadir, workdir):
        """Stack Sentinel-1 scenes together.

        :param Path datadir: Directory path with downloaded Sentinel-1 BURST data.
        :param Path workdir: Directory path where future computed data will be stored (cannot be the same as datadir).
        :return: None
        """
        if datadir == workdir:
            raise ValueError(
                f'Safety Triggered: datadir and workdir are the same location ({datadir}). '
                'Aborting to prevent accidental data deletion.'
            )
        if workdir in datadir.parents:
            raise ValueError(
                f'Safety Triggered: workdir ({workdir}) is a parent of datadir ({datadir}). '
                'Aborting to prevent accidental data deletion.'
            )

        # Reset generated files before the recursive scan: workdir may be
        # inside datadir, and stale reframed scenes would become input scenes.
        self.sbas = Stack(workdir, drop_if_exists=True)
        s1 = self._filter_excluded_dates(S1.scan_slc(datadir))
        self.logger.info('Stacking Sentinel-1 BURST data together.')
        self.sbas = self.sbas.set_scenes(s1)

    def _filter_excluded_dates(self, scenes):
        """Exclude all scenes on configured calendar dates, preserving source files."""
        excluded = pd.to_datetime(
            self._config.get('excluded_dates', []), errors='raise'
        ).normalize()
        if excluded.isna().any():
            raise ValueError('excluded_dates contains an empty or invalid date')
        mask = pd.to_datetime(scenes.index).normalize().isin(excluded)
        if mask.any():
            removed = sorted(set(pd.to_datetime(scenes.index[mask]).strftime('%Y-%m-%d')))
            self.logger.info(f'Excluding {int(mask.sum())} scenes on dates: {removed}')
        filtered = scenes.loc[~mask].copy()
        if filtered.empty:
            raise ValueError('No Sentinel-1 scenes remain after applying excluded_dates')
        return filtered

    def _reframe_scenes(self, aoi):
        """Reframe stacked Sentinel-1 data to smaller area of interest and stitch them together.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :return: None
        """
        self.logger.info('Reframing Sentinel-1 data.')
        self.sbas.compute_reframe(aoi)

    def _load_dem_and_landmask(self, aoi, dem, landmask):
        """Load DEM and landmask to stacked and reframed Sentinel-1 data.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path dem: Path to DEM file.
        :param Path landmask: Path to landmask file.
        :return: None
        """
        self.logger.info('Loading DEM and landmask to reframed Sentinel-1 data.')
        self.sbas.load_dem(str(dem), aoi)
        self.sbas.load_landmask(str(landmask))

    def _align_images(self):
        """Align Sentinel-1 images.

        :return: None
        """
        self.logger.info('Aligning Sentinel-1 data.')
        self.sbas.compute_align()

    def _geocoding_transform(self, coarsen=10.0):
        """Geocode Sentinel-1 images.

        :param float coarsen: Downsampling factor used to control the output pixel size.
                              A higher value results in faster processing and smaller files but lower spatial detail. Defaults to 10.
        :return: None
        """
        self.logger.info('Geocoding Sentinel-1 data.')
        self.sbas.compute_geocode(coarsen=coarsen)

    def _find_optimal_network(
        self, basedays=(36, 48, 60, 80), basemeters=(80, 100, 120, 150)
    ):
        """Analyzes all possible scene combinations to find the optimal fully connected SBAS network.

        :param tuple basedays: Possible values for maximum temporal baseline (days). Defaults to (36, 48, 60, 80).
        :param tuple basemeters: Possible values for maximum perpendicular baseline (m). Defaults to (80, 100, 120, 150).
        :return: None
        """
        stack_df = self.sbas.to_dataframe()
        all_dates = list(stack_df.index.astype(str))
        results = []

        self.logger.info('Finding optimal SBAS network.')
        for days in basedays:
            for meters in basemeters:
                # Build candidate pairs
                pairs_df = (
                    self.sbas.sbas_pairs(days=days, meters=meters)
                    if hasattr(self.sbas, 'sbas_pairs')
                    else self.sbas.baseline_pairs(days=days, meters=meters)
                )

                # Connectivity check
                adj = defaultdict(set)
                for a, b in pairs_df.iloc[:, :2].astype(str).values:
                    adj[a].add(b)
                    adj[b].add(a)

                seen, components = set(), 0
                for node in all_dates:
                    if node not in seen:
                        components += 1
                        stack = [node]
                        while stack:
                            curr = stack.pop()
                            if curr not in seen:
                                seen.add(curr)
                                stack.extend(adj[curr] - seen)

                results.append(
                    {
                        'days': days,
                        'meters': meters,
                        'n_pairs': len(pairs_df),
                        'n_comp': components,
                        'df': pairs_df,
                    }
                )

        valid = [r for r in results if r['n_comp'] == 1]
        if not valid:
            if self._diagnostics is not None:
                self._diagnostics.network(None, [{k: v for k, v in r.items() if k != 'df'} for r in results])
            raise RuntimeError('No connected network found. Try increasing thresholds.')

        best_config = min(valid, key=lambda x: (x['n_pairs'], x['days'], x['meters']))
        self.baseline_pairs = best_config['df']
        if self._diagnostics is not None:
            self._diagnostics.network(self.baseline_pairs,
                                      [{k: v for k, v in r.items() if k != 'df'} for r in results])

    def _compute_interferograms(
        self,
        intensity_wavelength=30,
        phase_wavelength=30,
        coarsen=(1, 4),
        goldstein_patch=8,
    ):
        """Compute interferograms from baseline pairs.

        :param int intensity_wavelength: Gaussian smoothing cut-off wavelength (metres) for intensity. Defaults to 30; must match phase_wavelength.
        :param int phase_wavelength: Gaussian smoothing cut-off wavelength (metres) for wrapped phase. Defaults to 30.
        :param tuple coarsen: Radar coordinate downsampling (range_factor, azimuth_factor). Defaults to (1, 4).
        :param int goldstein_patch: Window size (pixels) for Goldstein filtering. Defaults to 8.
        :return: None
        """
        if intensity_wavelength != phase_wavelength:
            raise ValueError('Intensity and phase must use the same averaging kernel for coherence.')
        topo = self.sbas.get_topo()
        data = self.sbas.open_data()
        self._check_slc_coverage(data)

        # Process Intensity (for correlation weights)
        intensity = self.sbas.multilooking(
            np.square(np.abs(data)), wavelength=intensity_wavelength, coarsen=coarsen
        )

        # Compute Phase Difference
        phase = self.sbas.phasediff(self.baseline_pairs, data, topo)
        phase = self.sbas.multilooking(
            phase, wavelength=phase_wavelength, coarsen=coarsen
        )

        # Filter & Synthesis
        self.corr = self.sbas.correlation(phase, intensity)
        phase_goldstein = self.sbas.goldstein(phase, self.corr, goldstein_patch)

        self.logger.info('Computing interferograms.')
        self.intf = self.sbas.interferogram(phase_goldstein)

    def _check_slc_coverage(self, data):
        """Reject empty aligned acquisitions before generating their interferograms."""
        counts = (np.isfinite(data) & (np.abs(data) > 0)).sum(('y', 'x')).compute()
        report = counts.to_dataframe(name='valid_slc_pixels')
        if self._diagnostics is not None:
            self._diagnostics.table('slc_coverage', report.reset_index())
        result_dir = (self._config or {}).get('result_dir')
        if result_dir is not None:
            quality_dir = Path(result_dir) / 'quality'
            quality_dir.mkdir(parents=True, exist_ok=True)
            report.to_csv(quality_dir / 'slc_coverage.csv')
        empty = report.index[report.valid_slc_pixels == 0]
        if len(empty):
            dates = ', '.join(pd.to_datetime(empty).strftime('%Y-%m-%d'))
            raise ValueError(
                f'Aligned SLC has no valid samples in the processing AOI on: {dates}. '
                'Check burst completeness and alignment, or explicitly exclude the affected dates. '
                'See quality/slc_coverage.csv when result_dir is configured.'
            )

    def _unwrap_phase(self, corr_limit=0.20, unwrap_m=10.0):
        """Unwrap phases using SNAPHU.

        :param float corr_limit: Minimum correlation threshold for masking. Defaults to 0.20.
        :param float unwrap_m: Target spatial resolution for unwrapping in meters. Defaults to 10.0.
        :return: None
        """
        if self.intf is None or self.corr is None:
            raise RuntimeError(
                'Interferograms and correlation must be computed before unwrapping.'
            )

        # Decimate to target unwrap spacing
        dec_u = self.sbas.decimator(unwrap_m)
        # Wrapped angles must be averaged on the unit circle, not arithmetically.
        corr_u, phasor_u = dask.persist(
            dec_u(self.corr), dec_u(np.exp(1j * self.intf))
        )
        intf_u = xr.apply_ufunc(np.angle, phasor_u, dask='allowed')

        # Build Correlation Mask
        valid = np.isfinite(corr_u) & (corr_u >= corr_limit) & np.isfinite(intf_u)
        corr_mask = corr_u.where(valid)
        self.corr_unwrap = corr_mask
        if self._diagnostics is not None:
            self._diagnostics.summary['unwrapping'] = {'coherence_threshold': corr_limit,
                                                       'target_spacing_m': unwrap_m}
            self._diagnostics.save()
            self._diagnostics.pairs(intf_u.where(valid), corr_mask, 'wrapped_phase')

        # Verify we have valid pixels to unwrap
        n_valid = int(np.isfinite(corr_mask).sum().compute())
        if n_valid == 0:
            raise RuntimeError(
                f'No pixels found above corr_limit={corr_limit}. Unwrapping aborted.'
            )

        self.logger.info('Unwrapping phases.')
        self.unwrap = self.sbas.unwrap_snaphu(
            intf_u.where(valid), corr_mask
        ).persist()

        # Trigger computation to catch SNAPHU execution errors immediately
        self.unwrap = self.unwrap.compute()
        self.unwrap['phase'] = self.unwrap.phase.where(valid)
        if self._diagnostics is not None:
            self._diagnostics.pairs(self.unwrap.phase, corr_mask, 'unwrapped_phase')
        if not bool(np.isfinite(self.unwrap.phase).any()):
            raise RuntimeError('SNAPHU returned no valid unwrapped phase.')

    def _detrend_phase(self, chunksize=256):
        """Preserve deformation and optionally subtract a stable-area pair offset.

        A local Gaussian high-pass removes embankment-scale deformation along
        with atmosphere. No spatial trend is fitted without stable-ground data.
        The historical method name is retained for pipeline callers.
        """
        if self.unwrap is None:
            raise RuntimeError('Phase must be unwrapped before detrending.')

        # Ensure Dask chunking
        phase_data = self.unwrap.phase
        chunk_spec = {d: chunksize for d in phase_data.dims if d in ('y', 'x')}
        if chunk_spec:
            phase_data = phase_data.chunk(chunk_spec)

        reference = (self._config or {}).get('reference_area_wkt', '')
        if reference:
            from shapely import contains_xy

            polygon = loads(reference)
            if not isinstance(polygon, (Polygon, MultiPolygon)) or polygon.is_empty or not polygon.is_valid:
                raise ValueError('reference_area_wkt must be a valid nonempty Polygon or MultiPolygon.')
            aoi = (self._config or {}).get('aoi')
            if aoi is not None and not aoi.covers(polygon):
                raise ValueError('Stable reference polygon must be inside the processing AOI.')
            # PyGMTSAR.geocode supports Polygon, but not MultiPolygon.
            polygons = list(polygon.geoms) if isinstance(polygon, MultiPolygon) else [polygon]
            xx, yy = np.meshgrid(phase_data.x.values, phase_data.y.values)
            reference_pixels = np.zeros(xx.shape, dtype=bool)
            for part in polygons:
                reference_pixels |= contains_xy(self.sbas.geocode(part), xx, yy)
            mask = xr.DataArray(reference_pixels,
                                dims=('y', 'x'), coords={'y': phase_data.y, 'x': phase_data.x})
            reference_phase = phase_data.where(mask)
            diagnostics = xr.Dataset({
                'reference_valid_phase_pixels': reference_phase.count(('y', 'x')),
                'all_valid_phase_pixels': phase_data.count(('y', 'x')),
                'reference_mean_phase_rad': reference_phase.mean(('y', 'x')),
            })
            if self.corr_unwrap is not None:
                _, reference_corr = xr.align(phase_data, self.corr_unwrap, join='exact')
                diagnostics['reference_coherent_pixels'] = reference_corr.where(mask).count(('y', 'x'))
            report = diagnostics.compute().to_dataframe()
            report['reference_mask_pixels'] = int(reference_pixels.sum())
            if self._diagnostics is not None:
                self._diagnostics.reference(report)
            result_dir = (self._config or {}).get('result_dir')
            if result_dir is not None:
                quality_dir = Path(result_dir) / 'quality'
                quality_dir.mkdir(parents=True, exist_ok=True)
                report.to_csv(quality_dir / 'reference_phase_diagnostics.csv')
            missing = report['reference_valid_phase_pixels'] == 0
            if missing.any():
                failed_pairs = report.index[missing].astype(str).tolist()
                raise ValueError(
                    f'Stable reference area has no valid phase for {int(missing.sum())} '
                    f'of {len(report)} pairs; reference mask has {int(reference_pixels.sum())} pixels. '
                    f'Affected pairs: {", ".join(failed_pairs)}. '
                    'See quality/reference_phase_diagnostics.csv when result_dir is configured.'
                )
            offset = reference_phase.mean(('y', 'x'), skipna=True)
            phase_data = phase_data - offset
        else:
            self.logger.warning('No stable-ground reference configured; preserving unwrapped phase without spatial detrending.')
        self.detrend = phase_data.persist()

        # Trigger computation
        _ = float(self.detrend.isel(pair=0).mean().compute())

    def _compute_displacement(self, target_m=10.0):
        """Compute cumulative LOS displacement and error metrics.

        :param float target_m: Target spatial resolution. Defaults to 10.0.
        :return: None
        """
        if self.detrend is None or self.corr is None:
            raise RuntimeError('Missing detrended phase or correlation data.')

        # Grid Alignment & SBAS Solve
        corr_ra = self.corr_unwrap
        if corr_ra is None:
            corr_ra = self.sbas.decimator(float(target_m))(self.corr).persist()
        # Matching shapes alone does not establish matching pixel coordinates.
        phase_ra, corr_ra = xr.align(self.detrend, corr_ra, join='exact')
        corr_ra = corr_ra.where(np.isfinite(phase_ra))

        with ProgressBar():
            sol = self.sbas.lstsq(phase_ra, corr_ra)
            # PyGMTSAR's least-squares solver returns minimum-norm values even
            # when masking disconnects a pixel's temporal network. Those dates
            # have no displacement relative to the first epoch and are nodata.
            pairs, dates = self.sbas.get_pairs(phase_ra, dates=True)
            dates = pd.to_datetime(dates)
            refs = dates.get_indexer(pd.to_datetime(pairs.ref))
            reps = dates.get_indexer(pd.to_datetime(pairs.rep))
            connected = xr.apply_ufunc(
                self._reference_connected,
                (np.isfinite(phase_ra) & np.isfinite(corr_ra) & (corr_ra > 0)).chunk({'pair': -1}),
                input_core_dims=[['pair']], output_core_dims=[['date']],
                kwargs={'refs': refs, 'reps': reps, 'n_dates': len(dates)},
                vectorize=True, dask='parallelized', output_dtypes=[bool],
                dask_gufunc_kwargs={'output_sizes': {'date': len(dates)}},
            ).assign_coords(date=dates)
            sol = sol.where(connected)
            disp_ra = self.sbas.los_displacement_mm(sol).persist()

        # Reference to Zero (Inline Time-Dim Selection)
        t_dim = next(
            (d for d in disp_ra.dims if d in ('date', 'time', 'epoch', 'pair')), None
        )
        if t_dim:
            disp_ra = disp_ra - disp_ra.isel({t_dim: 0})
        vel_ra = self.sbas.velocity(disp_ra).persist()

        # Geocoding & Coordinate Transformation
        self.sbas.compute_geocode(float(target_m))
        self.logger.info('Computing displacements.')
        self.disp_ll = self.sbas.cropna(self.sbas.ra2ll(disp_ra)).persist()
        self.vel_ll = self.sbas.cropna(self.sbas.ra2ll(vel_ra)).persist()
        self.disp_ll.attrs.update(units='mm', spatial_reference='stable_area' if
                                 (self._config or {}).get('reference_area_wkt') else 'unreferenced')
        self.vel_ll.attrs['units'] = 'mm/year'

        # RMSE Calculation
        disp_pairs_ra = self.sbas.los_displacement_mm(self.detrend).persist()
        # PyGMTSAR.rmse wraps residuals to [-pi, pi]; that is unsuitable for mm.
        rep_disp = disp_ra.sel(date=phase_ra.rep).drop_vars('date')
        ref_disp = disp_ra.sel(date=phase_ra.ref).drop_vars('date')
        residual = disp_pairs_ra - (rep_disp - ref_disp)
        weights = corr_ra.where(np.isfinite(residual))
        weight_sum = weights.sum('pair')
        self.rmse = np.sqrt(
            (weights * residual**2).sum('pair', min_count=1)
            / weight_sum.where(weight_sum > 0)
        ).persist()
        self.quality_ll = {
            'pair_rmse_mm': self.sbas.ra2ll(self.rmse),
            'mean_coherence': self.sbas.ra2ll(self.corr.mean('pair')),
            'valid_pair_fraction': self.sbas.ra2ll(np.isfinite(phase_ra).mean('pair')),
        }
        if self._diagnostics is not None:
            self._diagnostics.displacement(self.disp_ll)

    @staticmethod
    def _reference_connected(valid, refs, reps, n_dates):
        """Find acquisition dates connected to the first epoch at one pixel."""
        parents = list(range(n_dates))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for a, b in zip(refs[valid], reps[valid]):
            parents[root(b)] = root(a)
        first = root(0)
        connected = np.array([root(i) == first for i in range(n_dates)])
        # An isolated reference epoch is not an observed zero-displacement pixel.
        if connected.sum() == 1:
            connected[:] = False
        return connected

    def _compute_risk(self):
        """Compute risk map based on displacements, velocity and slope.

        :return: None
        """
        if self.disp_ll is None:
            raise RuntimeError(
                'Missing displacement data. Run _compute_displacement first.'
            )

        self.logger.info('Computing risk map..')
        # Setup Data Layers
        disp = self.disp_ll
        t_dim = next((d for d in disp.dims if d in ('date', 'time', 'epoch')), 'date')

        # Velocity (mm/day)
        vel_map = np.abs(self.sbas.velocity(disp))

        # Recent Change (dlos) - Difference between last two acquisitions
        dlos_map = np.abs(disp.isel({t_dim: -1}) - disp.isel({t_dim: -2}))

        # Cumulative Displacement (los)
        max_disp = np.abs(disp.isel({t_dim: -1}))

        # Extract Slope from DEM
        try:
            temp_disp = disp.copy(deep=True).rio.write_crs('EPSG:4326')

            # Standardize the template dimensions to 'x' and 'y' for the reprojection engine
            if 'lon' in temp_disp.dims and 'lat' in temp_disp.dims:
                temp_disp = temp_disp.rename({'lon': 'x', 'lat': 'y'})

            temp_disp = temp_disp.rio.set_spatial_dims(x_dim='x', y_dim='y')

            # Fetch and Prepare DEM
            dem = self.sbas.get_dem()
            if dem.rio.crs is None:
                dem = dem.rio.write_crs('EPSG:4326')

            # Standardize DEM dims as well
            d_x = (
                'lon'
                if 'lon' in dem.dims
                else ('longitude' if 'longitude' in dem.dims else 'x')
            )
            d_y = (
                'lat'
                if 'lat' in dem.dims
                else ('latitude' if 'latitude' in dem.dims else 'y')
            )
            dem = dem.rename({d_x: 'x', d_y: 'y'}).rio.set_spatial_dims(
                x_dim='x', y_dim='y'
            )

            # Match DEM to InSAR grid
            dem_match = dem.rio.reproject_match(temp_disp).squeeze(drop=True)

            # Extract Coordinates for Gradient
            y_coords = dem_match.coords['y'].values
            x_coords = dem_match.coords['x'].values

            # Geodetic distance approximation (meters)
            lat0 = np.nanmean(y_coords)
            dy = abs(np.nanmedian(np.diff(y_coords))) * 111320.0
            dx = (
                abs(np.nanmedian(np.diff(x_coords)))
                * 111320.0
                * np.cos(np.deg2rad(lat0))
            )

            # Calculate slope on the underlying values
            arr = dem_match.ffill('x').bfill('x').values
            dz_dy, dz_dx = np.gradient(arr, dy, dx)
            slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

            sample = disp.isel({t_dim: 0}, drop=True)
            slope_da = xr.DataArray(
                slope_deg, coords=sample.coords, dims=sample.dims, name='slope'
            )

        except Exception as e:
            self.logger.error(f'[RISK] Slope extraction failed: {e}')
            sample = disp.isel({t_dim: 0}, drop=True)
            slope_da = xr.DataArray(
                np.zeros(sample.shape), coords=sample.coords, dims=sample.dims
            )

        risk_score = xr.where(vel_map >= 2.0, 3, 0)  # |vel| >= 2 mm/day
        risk_score += xr.where(dlos_map >= 10.0, 3, 0)  # |dlos| >= 10 mm
        risk_score += xr.where(max_disp >= 50.0, 2, 0)  # |los| >= 50 mm
        risk_score += xr.where(slope_da >= 15.0, 1, 0)  # slope >= 15°

        self.risk_map = risk_score.rename('risk_score').persist()
        self.failure_flag = (
            xr.where(self.risk_map >= 6, 1, 0).rename('failure_flag').persist()
        )

    def _environmental_database(
        self, aoi, output_dir, grid_rows=3, grid_cols=3, model='era5'
    ):
        """Extracts daily climate and air quality data for the provided AOI.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_dir: Directory path where final results will be saved.
        :param int grid_rows: Number of horizontal sampling points across the AOI. Defaults to 3.
        :param int grid_cols: Number of vertical sampling points across the AOI. Defaults to 3.
        :param str model: The atmospheric reanalysis model to use for historical climate data. Defaults to "era5".
        :return: None
        """
        self.logger.info('Creating environmental database.')
        # Temporal & Spatial Setup
        stack_df = self.sbas.to_dataframe()
        dt_index = pd.to_datetime(stack_df.index)
        start_date, stop_date = dt_index.min().date(), dt_index.max().date()

        aoi = loads(aoi) if isinstance(aoi, str) else aoi
        bounds = aoi.bounds
        center_lon, center_lat = (
            (bounds[0] + bounds[2]) / 2,
            (bounds[1] + bounds[3]) / 2,
        )

        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=center_lon, lat=center_lat) or 'UTC'

        # API Client Setup
        cache_session = requests_cache.CachedSession(
            str(output_dir / 'openmeteo_cache'), expire_after=3600
        )
        retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
        om_client = openmeteo_requests.Client(session=retry_session)

        # Build Grid & Sample Altitudes
        lats, lons = (
            np.linspace(bounds[1], bounds[3], grid_rows),
            np.linspace(bounds[0], bounds[2], grid_cols),
        )
        locations = {'center': {'lat': center_lat, 'lon': center_lon}}
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                locations[f'grid_{i}_{j}'] = {'lat': float(lat), 'lon': float(lon)}

        try:
            dem = self.sbas.get_dem()
            for k, v in locations.items():
                try:
                    val = dem.sel(lat=v['lat'], lon=v['lon'], method='nearest').values
                    locations[k]['alt'] = float(np.nan_to_num(val))
                except Exception:
                    locations[k]['alt'] = 0
        except Exception as e:
            self.logger.error(f'[ENVDB] Altitude sampling skipped: {e}')

        # Fetch & Process Climate Data
        climate_url = 'https://archive-api.open-meteo.com/v1/archive'
        climate_vars = [
            'temperature_2m_max',
            'precipitation_sum',
            'et0_fao_evapotranspiration',
            'wind_gusts_10m_max',
        ]
        all_climate = []

        for name, loc in locations.items():
            params = {
                'latitude': loc['lat'],
                'longitude': loc['lon'],
                'start_date': start_date.isoformat(),
                'end_date': stop_date.isoformat(),
                'daily': climate_vars,
                'timezone': tz_name,
                'models': model,
            }
            res = om_client.weather_api(climate_url, params=params)[0]
            daily = res.Daily()

            num_days = len(daily.Variables(0).ValuesAsNumpy())
            dates = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit='s', utc=True)
                .tz_convert(tz_name)
                .tz_localize(None),
                periods=num_days,
                freq='D',
            )

            df = pd.DataFrame({'date': dates, 'location': name})
            for idx, var in enumerate(climate_vars):
                df[var] = daily.Variables(idx).ValuesAsNumpy()

            # Rainfall indicators
            df['precip_7d_mm'] = df['precipitation_sum'].rolling(7, min_periods=1).sum()
            df['precip_30d_mm'] = (
                df['precipitation_sum'].rolling(30, min_periods=1).sum()
            )
            df['wet_days_30d'] = (
                (df['precipitation_sum'] > 1.0).rolling(30, min_periods=1).sum()
            )

            # Hydrological balance (Precip - Evapotranspiration)
            df['daily_wb'] = df['precipitation_sum'] - df['et0_fao_evapotranspiration']
            df['water_balance_30d_mm'] = df['daily_wb'].rolling(30, min_periods=1).sum()

            # Wind Gusts
            df['wind_gust_kmh'] = df['wind_gusts_10m_max']

            # Temperature Anomaly (Current Max vs 30-day average)
            df['temp_max_C_month_anom'] = (
                df['temperature_2m_max']
                - df['temperature_2m_max'].rolling(30, min_periods=1).mean()
            )

            all_climate.append(df)

        climate_df = pd.concat(all_climate).reset_index(drop=True)

        # Fetch & Process Air Quality Data
        aq_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        aq_vars = ['pm10', 'pm2_5']
        all_aq = []

        # Helper: US-AQI calculation for PM2.5
        def calculate_aqi(pm25):
            if pm25 < 0:
                return 0
            if pm25 <= 12.0:
                return ((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0
            if pm25 <= 35.4:
                return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
            if pm25 <= 55.4:
                return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
            return 200  # Cap for simplified model

        for name, loc in locations.items():
            params = {
                'latitude': loc['lat'],
                'longitude': loc['lon'],
                'start_date': start_date.isoformat(),
                'end_date': stop_date.isoformat(),
                'hourly': aq_vars,
                'timezone': tz_name,
            }
            res = om_client.weather_api(aq_url, params=params)[0]
            hourly = res.Hourly()

            dates = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit='s', utc=True)
                .tz_convert(tz_name)
                .tz_localize(None),
                periods=len(hourly.Variables(0).ValuesAsNumpy()),
                freq='h',
            )

            df_h = pd.DataFrame({'datetime': dates})
            for idx, var in enumerate(aq_vars):
                df_h[var] = hourly.Variables(idx).ValuesAsNumpy()

            # Resample to daily max and calculate AQI
            df_aq = df_h.resample('D', on='datetime').max().reset_index()
            df_aq['us_aqi'] = df_aq['pm2_5'].apply(calculate_aqi)
            df_aq['location'] = name
            # Fix column name for resample consistency
            df_aq = df_aq.rename(columns={'datetime': 'date'})
            all_aq.append(df_aq)

        air_df = pd.concat(all_aq).reset_index(drop=True)

        # Save and Manifest
        climate_df.to_csv(output_dir / 'climate_daily_db.csv', index=False)
        air_df.to_csv(output_dir / 'air_quality_daily_db.csv', index=False)

        manifest = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'detected_timezone': tz_name,
            'date_range': [start_date.isoformat(), stop_date.isoformat()],
            'locations': locations,
        }
        with open(output_dir / 'envdb_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

    def _compute_risk_database(self, output_dir):
        """Merges InSAR monitoring data with environmental data and computes
        a multi-hazard composite risk score for every measurement point.

        :param Path output_dir: Directory path where final results will be saved and 'climate_daily_db.csv'
                                and 'air_quality_daily_db.csv' are already stored.
        :return: None
        """
        # Extract and Merge InSAR Layers
        if self.disp_ll is None or self.risk_map is None:
            raise RuntimeError(
                'InSAR displacement or risk map not found. Run previous steps first.'
            )

        self.logger.info('Computing risk database.')
        # Convert xarray datasets to long-form DataFrames
        df_disp = self.disp_ll.to_dataframe().reset_index()
        df_risk = self.risk_map.to_dataframe().reset_index()

        if 'date' in df_risk.columns:
            df_risk = df_risk.drop(columns=['date'])

        # Merge on spatial coordinates (lat, lon)
        df_base = df_disp.merge(df_risk, on=['lat', 'lon'])

        # Normalize and Calculate Missing Motion Columns
        rename_map = {
            'displacement': 'los_mm',
            'los': 'los_mm',
            'slope': 'slope_deg',
            'risk_score': 'insar_risk_score_base',
        }
        df_base = df_base.rename(
            columns={k: v for k, v in rename_map.items() if k in df_base.columns}
        )

        # Suffix Fallback: If 'date' was somehow still renamed by pandas
        if 'date' not in df_base.columns:
            if 'date_x' in df_base.columns:
                df_base = df_base.rename(columns={'date_x': 'date'})
            elif 'date_y' in df_base.columns:
                df_base = df_base.rename(columns={'date_y': 'date'})

        # Ensure cell_id exists for time-series grouping
        df_base['cell_id'] = (
            df_base['lat'].astype(str) + '_' + df_base['lon'].astype(str)
        )
        df_base['date'] = pd.to_datetime(df_base['date']).dt.normalize()

        # Calculate dlos (daily change) and velocity (rate)
        df_base = df_base.sort_values(['cell_id', 'date'])
        df_base['dlos_mm'] = df_base.groupby('cell_id')['los_mm'].diff().fillna(0)

        # Calculate days between acquisitions to get true velocity
        date_diff = df_base.groupby('cell_id')['date'].diff().dt.days.fillna(1)
        date_diff = date_diff.replace(0, 1)  # Prevent division by zero
        df_base['vel_mm_per_day'] = (df_base['dlos_mm'] / date_diff).abs()

        # Load Environmental Data
        clim = pd.read_csv(output_dir / 'climate_daily_db.csv')
        air = pd.read_csv(output_dir / 'air_quality_daily_db.csv')

        clim['date'] = pd.to_datetime(clim['date']).dt.normalize()
        air['date'] = pd.to_datetime(air['date']).dt.normalize()

        # Risk Thresholds and Weights
        th = {
            'VEL_DAY_LO': 0.5,
            'VEL_DAY_HI': 2.0,
            'DLOS_DAY_LO': 0.5,
            'DLOS_DAY_HI': 2.0,
            'LOS_LO': 20.0,
            'LOS_HI': 60.0,
            'SLOPE_LO': 10.0,
            'SLOPE_HI': 20.0,
            'P7_LO': 20.0,
            'P7_HI': 60.0,
            'P30_LO': 60.0,
            'P30_HI': 150.0,
            'PM25_LO': 15.0,
            'PM25_HI': 35.0,
            'RISK_MODERATE': 35.0,
            'RISK_HIGH': 60.0,
            'RISK_CRITICAL': 80.0,
        }
        w = {'motion': 0.45, 'terrain': 0.15, 'hydroclimate': 0.25, 'atmosphere': 0.15}

        def _lin_score(x, lo, hi):
            return np.clip(
                (np.asarray(x, dtype='float64') - lo) / (hi - lo + 1e-12), 0.0, 1.0
            )

        # Spatial Matching (Robust Nearest Neighbor)
        cell_lut = df_base[['cell_id', 'lat', 'lon']].drop_duplicates('cell_id')

        # Check if environmental data has coordinates for distance matching
        has_env_coords = 'lat' in clim.columns and 'lon' in clim.columns

        if has_env_coords:
            env_pts = clim[['location', 'lat', 'lon']].drop_duplicates('location')
            p_lat, p_lon = env_pts['lat'].values, env_pts['lon'].values
            c_lat, c_lon = cell_lut['lat'].values, cell_lut['lon'].values

            dist = (c_lat[:, None] - p_lat[None, :]) ** 2 + (
                c_lon[:, None] - p_lon[None, :]
            ) ** 2
            cell_lut['env_location'] = env_pts['location'].values[dist.argmin(axis=1)]
        else:
            # Fallback: Map to the first location name if coordinates are missing
            primary_loc = clim['location'].iloc[0]
            cell_lut['env_location'] = primary_loc

        # Merge Databases
        df_merged = df_base.merge(cell_lut[['cell_id', 'env_location']], on='cell_id')

        # Prefix columns to avoid collisions
        clim_cols = {
            c: f'climate_{c}' for c in clim.columns if c not in ['date', 'location']
        }
        air_cols = {c: f'air_{c}' for c in air.columns if c not in ['date', 'location']}

        df_merged = df_merged.merge(
            clim.rename(columns=clim_cols),
            left_on=['date', 'env_location'],
            right_on=['date', 'location'],
            how='left',
        )
        df_merged = df_merged.merge(
            air.rename(columns=air_cols),
            left_on=['date', 'env_location'],
            right_on=['date', 'location'],
            how='left',
        )

        # Compute Composite Risk Score
        # Motion
        v_s = _lin_score(
            df_merged['vel_mm_per_day'], th['VEL_DAY_LO'], th['VEL_DAY_HI']
        )
        d_s = _lin_score(
            np.abs(df_merged['dlos_mm']), th['DLOS_DAY_LO'], th['DLOS_DAY_HI']
        )
        l_s = _lin_score(np.abs(df_merged['los_mm']), th['LOS_LO'], th['LOS_HI'])
        motion_comp = 0.40 * v_s + 0.35 * d_s + 0.25 * l_s

        # Other Components
        hydro_comp = _lin_score(
            df_merged.get('climate_precip_7d_mm', 0), th['P7_LO'], th['P7_HI']
        )
        terrain_comp = _lin_score(
            df_merged.get('slope_deg', 0), th['SLOPE_LO'], th['SLOPE_HI']
        )
        atmos_comp = _lin_score(
            df_merged.get('air_pm2_5', 0), th['PM25_LO'], th['PM25_HI']
        )

        # Final Risk Score (0-100)
        total_risk = 100.0 * (
            w['motion'] * motion_comp
            + w['terrain'] * terrain_comp
            + w['hydroclimate'] * hydro_comp
            + w['atmosphere'] * atmos_comp
        )

        df_merged['risk_score_0to100'] = total_risk.astype('float32')

        # Classification
        level = np.full(len(df_merged), 'Low', dtype=object)
        level[total_risk >= th['RISK_MODERATE']] = 'Moderate'
        level[total_risk >= th['RISK_HIGH']] = 'High'
        level[total_risk >= th['RISK_CRITICAL']] = 'Critical'
        df_merged['risk_class'] = pd.Categorical(
            level, categories=['Low', 'Moderate', 'High', 'Critical'], ordered=True
        )

        # UTM Projection
        mean_lat, mean_lon = df_merged['lat'].mean(), df_merged['lon'].mean()
        utm_zone = int(np.floor((mean_lon + 180.0) / 6.0) + 1)
        epsg = 32700 + utm_zone if mean_lat < 0 else 32600 + utm_zone

        transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
        xe, yn = transformer.transform(df_merged['lon'].values, df_merged['lat'].values)
        df_merged['UTM_E'], df_merged['UTM_N'] = xe, yn

        # Export
        df_merged.to_csv(output_dir / 'final_risk_database.csv', index=False)

        # Save Metadata
        governance = {
            'calculated_at': datetime.now(timezone.utc).isoformat(),
            'thresholds': th,
            'weights': w,
            'utm_epsg': int(epsg),
        }
        with open(output_dir / 'risk_governance.json', 'w') as f:
            json.dump(governance, f, indent=2)

    def _export_displacements(self, output_dir):
        """Export displacement time-series and velocity maps.

        :param Path output_dir: Base output directory.
        :return: None
        """
        base_path = Path(output_dir)
        disp_path = base_path / 'displacements'
        vel_path = base_path / 'velocity'

        disp_path.mkdir(parents=True, exist_ok=True)
        vel_path.mkdir(parents=True, exist_ok=True)

        quality_path = base_path / 'quality'
        quality_path.mkdir(parents=True, exist_ok=True)
        for name, data in getattr(self, 'quality_ll', {}).items():
            grid = data.rename({'lat': 'y', 'lon': 'x'})
            grid.rio.write_crs('EPSG:4326', inplace=True)
            grid.rio.write_nodata(np.nan, inplace=True)
            grid.rio.to_raster(quality_path / f'{name}.tif')

        self.logger.info('Exporting displacements and velocity.')
        if hasattr(self, 'vel_ll') and self.vel_ll is not None:
            vel_to_export = self.vel_ll.rename({'lat': 'y', 'lon': 'x'})

            if vel_to_export.rio.crs is None:
                vel_to_export.rio.write_crs('EPSG:4326', inplace=True)

            vel_filename = vel_path / 'velocity.tif'
            vel_to_export.rio.write_nodata(np.nan, inplace=True)
            vel_to_export.rio.to_raster(vel_filename)

        if self.disp_ll is not None:
            t_dim = next(
                (
                    d
                    for d in self.disp_ll.dims
                    if d in ('date', 'time', 'epoch', 'pair')
                ),
                None,
            )

            if not t_dim:
                raise ValueError(
                    'Could not find a valid time/date dimension in self.disp_ll.'
                )

            data_to_export = self.disp_ll.rename({'lat': 'y', 'lon': 'x'})
            num_dates = len(data_to_export[t_dim])

            for i in range(num_dates):
                slice_data = data_to_export.isel({t_dim: i})

                date_val = pd.to_datetime(slice_data[t_dim].values)
                date_str = date_val.strftime('%Y%m%d')
                filename = disp_path / f'disp_{date_str}.tif'

                if slice_data.rio.crs is None:
                    slice_data.rio.write_crs('EPSG:4326', inplace=True)

                slice_data.rio.write_nodata(np.nan, inplace=True)
                slice_data.rio.to_raster(filename)

    def _cleanup(self, workdir):
        """Remove unnecessary directory with files after computation is done.

        :param Path workdir: Directory path where interim results were stored.
        :return: None
        """
        if os.path.exists(workdir) and os.path.isdir(workdir):
            shutil.rmtree(workdir)

    def _run_stage(self, method, *args, **kwargs):
        if self._diagnostics is None:
            return method(*args, **kwargs)
        record = {'name': method.__name__, 'status': 'running',
                  'started_utc': datetime.now(timezone.utc).isoformat()}
        self._diagnostics.summary['stages'].append(record)
        self._diagnostics.save()
        start = time.monotonic()
        try:
            result = method(*args, **kwargs)
            record['status'] = 'complete'
            return result
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            record['elapsed_seconds'] = round(time.monotonic() - start, 3)
            self._diagnostics.save()

    def _run(self):
        if self._config.get('diagnostics', True):
            root = self._config.get('diagnostics_root') or self._config['result_dir']
            self._diagnostics = InSARDiagnostics(root, self._config)
            self.logger.info(f'Intermediate diagnostics: {self._diagnostics.path}')
        try:
            result = self._run_processing()
        except Exception as exc:
            if self._diagnostics is not None:
                self._diagnostics.summary.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            if self._diagnostics is not None:
                self._diagnostics.summary['status'] = 'complete'
            return result
        finally:
            if self._diagnostics is not None:
                self._diagnostics.summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
                self._diagnostics.save()
            self.close()

    def _run_processing(self):
        glob_config = SettingsReader()
        dask_kwargs = {
            'silence_logs': glob_config['dask_parameters']['silence_logs'],
            'n_workers': glob_config['dask_parameters']['n_workers'],
            'threads_per_worker': glob_config['dask_parameters']['threads_per_worker'],
            'memory_limit': glob_config['dask_parameters']['memory_limit'],
        }
        self.logger.debug(f'Dask parameters: {dask_kwargs}')

        start = time.time()

        step = self._run_stage
        step(self._download_orbits, self._config['datadir'])
        step(self._download_dem, self._config['aoi'], self._config['dem_path'])
        step(self._download_landmask, self._config['aoi'], self._config['landmask_path'])
        step(self._run_dask_cluster, **dask_kwargs)
        step(self._stack_scenes, self._config['datadir'], self._config['workdir'])
        step(self._reframe_scenes, self._config['aoi'])
        step(self._load_dem_and_landmask,
            self._config['aoi'], self._config['dem_path'], self._config['landmask_path']
        )
        step(self._align_images)
        step(self._geocoding_transform)
        step(self._find_optimal_network)
        step(self._compute_interferograms)
        step(self._unwrap_phase)
        step(self._detrend_phase)
        step(self._compute_displacement)
        # Save InSAR products before optional external environmental requests.
        step(self._export_displacements, self._config['result_dir'])
        step(self._compute_risk)
        step(self._environmental_database, self._config['aoi'], self._config['result_dir'])
        step(self._compute_risk_database, self._config['result_dir'])
        step(self._cleanup, self._config['workdir'])

        elapsed_minutes = (time.time() - start) / 60
        self.logger.info(f'Computation completed in {elapsed_minutes:.2f} minutes.')
