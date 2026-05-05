from .base import PreprocessingBasePipeline

from pathlib import Path
from pygmtsar import S1, Tiles, Stack
import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from collections import defaultdict
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401


class Sentinel1Pipeline(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
        'params': {
            'aoi': {
                dtype: str,
                'description': 'POLYGON wkt string, coordinates must be in WGS84 (EPSG:4326)',
            }
        },
    }

    def _configure(self):
        self.client = None
        self.s1 = None
        self.stack = None
        self.dem = None
        self.landmask = None
        self.sbas = None
        self.dem_masked = None
        self.baseline_pairs = None
        self.corr = None
        self.intf = None
        self.unwrap = None
        self.detrend = None
        self.disp_ll = None
        self.rmse = None
        self.risk_map = None
        self.failure_flag = None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _download_orbits(self, datadir):
        """Download precise orbit files for Sentinel-1 BURST data.

        :param str datadir: Path to directory with downloaded BURST data
        :return: None.
        """
        self.s1 = S1.scan_slc(datadir)
        S1.download_orbits(datadir, self.s1)

    def _download_dem(self, aoi, output_dem):
        """Download DEM.
        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_dem: Path to output DEM file.
        :return: None
        """
        if not output_dem.exists():
            Tiles().download_dem(aoi, filename=output_dem, skip_exist=True)
        self.dem = output_dem

    def _download_landmask(self, aoi, output_landmask):
        """Download landmask.
        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_landmask: Path to output landmask file.
        :return: None
        """
        if not output_landmask.exists():
            Tiles().download_landmask(
                aoi, filename=output_landmask, skip_exist=True
            ).fillna(0)
        self.landmask = output_landmask

    def _run_dask_cluster(self, **kwargs):
        """Run Dask Cluster Client for computation.

        :return: None
        """
        if self.client is None:
            self.client = Client(**kwargs)

    def _stack_scenes(self, datadir, workdir):
        """Stack Sentinel-1 scenes together.

        :param str datadir: Directory path with downloaded Sentinel-1 BURST data.
        :param str workdir: Directory path where future computed data will be stored (cannot be the same as datadir).
        :return: None
        """
        data_path = Path(datadir).resolve()
        work_path = Path(workdir).resolve()

        if data_path == work_path:
            raise ValueError(
                f'Safety Triggered: datadir and workdir are the same location ({data_path}). '
                'Aborting to prevent accidental data deletion.'
            )
        if work_path in data_path.parents:
            raise ValueError(
                f'Safety Triggered: workdir ({work_path}) is a parent of datadir ({data_path}). '
                'Aborting to prevent accidental data deletion.'
            )

        self.s1 = S1.scan_slc(datadir)
        self.sbas = Stack(workdir, drop_if_exists=True).set_scenes(self.s1)

    def _reframe_scenes(self, aoi):
        """Reframe stacked Sentinel-1 data to smaller area of interest and stitch them together.

        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :return: None
        """
        self.sbas.compute_reframe(aoi)

    def _load_dem_and_landmask(self, aoi):
        """Load DEM and landmask to stacked and reframed Sentinel-1 data.

        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :return: None
        """
        self.sbas.load_dem(str(self.dem), aoi)
        self.sbas.load_landmask(str(self.landmask))
        self.dem_masked = self.sbas.get_dem().where(
            self.sbas.get_landmask()
        )  # maybe not needed

    def _align_images(self):
        """Align Sentinel-1 images.

        :return: None
        """
        self.sbas.compute_align()

    def _geocoding_transform(self):
        """Geocode Sentinel-1 images.

        :return: None
        """
        self.sbas.compute_geocode(coarsen=10.0)

    def _find_optimal_network(self):
        """Analyzes all possible scene combinations to find the optimal fully connected SBAS network.

        :return: None
        """
        basedays = [36, 48, 60, 80]
        basemeters = [80, 100, 120, 150]

        stack_df = self.sbas.to_dataframe()
        all_dates = list(stack_df.index.astype(str))
        results = []

        for days in basedays:
            for meters in basemeters:
                # Build candidate pairs
                pairs_df = (
                    self.sbas.sbas_pairs(days=days, meters=meters)
                    if hasattr(self.sbas, 'sbas_pairs')
                    else self.sbas.baseline_pairs(days=days, meters=meters)
                )

                # Connectivity check (DFS)
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
            raise RuntimeError('No connected network found. Try increasing thresholds.')

        best_config = min(valid, key=lambda x: (x['n_pairs'], x['days'], x['meters']))
        self.baseline_pairs = best_config['df']

    def _compute_interferograms(self):
        """Compute interferograms from baseline pairs.

        :return: None
        """
        intensity_wavelength = 20  # metres
        phase_wavelength = 30  # metres
        coarsen = (1, 4)
        goldstein_patch = 8  # pixels

        # 1. Load Data
        topo = self.sbas.get_topo()
        data = self.sbas.open_data()

        # 2. Process Intensity (for correlation weights)
        intensity = self.sbas.multilooking(
            np.square(np.abs(data)), wavelength=intensity_wavelength, coarsen=coarsen
        )

        # 3. Compute Phase Difference
        phase = self.sbas.phasediff(self.baseline_pairs, data, topo)
        phase = self.sbas.multilooking(
            phase, wavelength=phase_wavelength, coarsen=coarsen
        )

        # 4. Filter & Synthesis
        self.corr = self.sbas.correlation(phase, intensity)
        phase_goldstein = self.sbas.goldstein(phase, self.corr, goldstein_patch)

        self.intf = self.sbas.interferogram(phase_goldstein)

    def _unwrap_interferograms(self, corr_limit=0.20, unwrap_m=10.0):
        """Unwrap interferograms using SNAPHU.

        :param float corr_limit: Minimum correlation threshold for masking.
        :param float unwrap_m: Target spatial resolution for unwrapping in meters.
        :return: None
        """
        if self.intf is None or self.corr is None:
            raise RuntimeError(
                'Interferograms and correlation must be computed before unwrapping.'
            )

        # 1. Decimate to target unwrap spacing
        dec_u = self.sbas.decimator(unwrap_m)
        corr_u, intf_u = dask.persist(dec_u(self.corr), dec_u(self.intf))

        # 2. Build Correlation Mask
        corr_mask = corr_u.where(corr_u >= corr_limit)

        # Verify we have valid pixels to unwrap
        n_valid = int(np.isfinite(corr_mask).sum().compute())
        if n_valid == 0:
            raise RuntimeError(
                f'No pixels found above corr_limit={corr_limit}. Unwrapping aborted.'
            )

        # 3. Run SNAPHU Unwrapping
        self.unwrap = self.sbas.unwrap_snaphu(
            intf_u.where(corr_mask), corr_mask
        ).persist()

        # Trigger computation to catch SNAPHU execution errors immediately
        _ = float(self.unwrap.phase.isel(pair=0).mean().compute())

    def _detrend_phase(self, ramp_factor=3.0):
        """Remove long-wavelength ramps from the unwrapped phase in radar geometry.

        :param float ramp_factor: Multiplier for the smoothing wavelength. Higher values remove only very broad trends.
        :return: None
        """
        if self.unwrap is None:
            raise RuntimeError('Phase must be unwrapped before detrending.')

        # 1. Determine spatial scales
        base_wavelength = 30
        ramp_wavelength = ramp_factor * base_wavelength

        # 2. Ensure Dask chunking
        chunksize = 256
        phase_data = self.unwrap.phase
        chunk_spec = {d: chunksize for d in phase_data.dims if d in ('y', 'x')}
        if chunk_spec:
            phase_data = phase_data.chunk(chunk_spec)

        # 3. Build and subtract the ramp
        ramp = self.sbas.gaussian(phase_data, wavelength=ramp_wavelength).persist()

        # Subtracting the ramp leaves only the localized displacement signal
        self.detrend = (phase_data - ramp).persist()

        # 4. Trigger computation
        _ = float(self.detrend.isel(pair=0).mean().compute())

    def _compute_displacement(self, target_m=10.0):
        """Compute cumulative LOS displacement and error metrics.

        :param float target_m: Target spatial resolution.
        :return: None
        """
        if self.detrend is None or self.corr is None:
            raise RuntimeError('Missing detrended phase or correlation data.')

        # 1. Grid Alignment & SBAS Solve
        corr_ra = self.corr
        if self.detrend.sizes != self.corr.sizes:
            corr_ra = self.sbas.decimator(float(target_m))(self.corr).persist()

        with ProgressBar():
            sol = self.sbas.lstsq(self.detrend, corr_ra)
            disp_ra = self.sbas.los_displacement_mm(sol).persist()

        # 2. Reference to Zero (Inline Time-Dim Selection)
        t_dim = next(
            (d for d in disp_ra.dims if d in ('date', 'time', 'epoch', 'pair')), None
        )
        if t_dim:
            disp_ra = disp_ra.where(
                disp_ra[t_dim] != disp_ra[t_dim].values[0], other=np.nan
            )

        # 3. Geocoding & Coordinate Transformation
        self.sbas.compute_geocode(float(target_m))
        self.disp_ll = self.sbas.cropna(
            self.sbas.ra2ll(disp_ra)
        ).compute()  # can be switched to .persist()

        # 4. RMSE Calculation
        disp_pairs_ra = self.sbas.los_displacement_mm(self.detrend).persist()
        self.rmse = self.sbas.rmse(disp_pairs_ra, disp_ra, corr_ra).persist()

    def _compute_risk(self):
        """Compute risk map.

        :return: None
        """
        if self.disp_ll is None:
            raise RuntimeError(
                'Missing displacement data. Run _compute_displacement first.'
            )

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
            print(f'[RISK] Slope extraction failed: {e}')
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

    def _run(self):
        # self._download_dem(self.aoi, ...)
        pass
