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
from shapely.geometry.base import BaseGeometry
import json
import pandas as pd
import requests_cache
import openmeteo_requests
from datetime import datetime, timezone
from timezonefinder import TimezoneFinder
from retry_requests import retry
from shapely.wkt import loads
from pyproj import Transformer


class Sentinel1Pipeline(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
        'params': {
            'datadir': {
                'dtype': Path,
                'description': 'Path to the directory with Sentinel-1 SLC BURST data',
            },
            'aoi': {
                'dtype': str | BaseGeometry,
                'description': 'POLYGON wkt string or Shapely Polygon, coordinates must be in WGS84 (EPSG:4326)',
            },
            'dem_path': {
                'dtype': Path,
                'description': 'Path where the DEM file will be downloaded and later used',
            },
            'landmask_path': {
                'dtype': Path,
                'description': 'Path where the landmask file will be downloaded and later used',
            },
            'workdir': {
                'dtype': Path,
                'description': "Path to the directory where computed data will be stored (cannot be the same as 'datadir')",
            },
            'result_dir': {
                'dtype': Path,
                'description': 'Path to the directory where final results will be stored',
            },
        },
    }

    def _configure(self):
        self.client = None
        self.sbas = None
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

        :param Path datadir: Path to the directory with Sentinel-1 SLC BURST data.
        :return: None.
        """
        if not any(datadir.glob('*.SAFE/')):
            raise FileNotFoundError(f"No '.SAFE' directories found in {datadir}.")
        s1 = S1.scan_slc(datadir)
        S1.download_orbits(datadir, s1)

    def _download_dem(self, aoi, output_dem):
        """Download DEM.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_dem: Path to output DEM file.
        :return: None
        """
        Tiles().download_dem(aoi, filename=output_dem, skip_exist=True)

    def _download_landmask(self, aoi, output_landmask):
        """Download landmask.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_landmask: Path to output landmask file.
        :return: None
        """
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

        s1 = S1.scan_slc(datadir)
        self.sbas = Stack(workdir, drop_if_exists=True).set_scenes(s1)

    def _reframe_scenes(self, aoi):
        """Reframe stacked Sentinel-1 data to smaller area of interest and stitch them together.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :return: None
        """
        self.sbas.compute_reframe(aoi)

    def _load_dem_and_landmask(self, aoi, dem, landmask):
        """Load DEM and landmask to stacked and reframed Sentinel-1 data.

        :param str | BaseGeometry aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path dem: Path to DEM file.
        :param Path landmask: Path to landmask file.
        :return: None
        """
        self.sbas.load_dem(str(dem), aoi)
        self.sbas.load_landmask(str(landmask))

    def _align_images(self):
        """Align Sentinel-1 images.

        :return: None
        """
        self.sbas.compute_align()

    def _geocoding_transform(self, coarsen=10.0):
        """Geocode Sentinel-1 images.

        :param float coarsen: Downsampling factor used to control the output pixel size.
                              A higher value results in faster processing and smaller files but lower spatial detail. Defaults to 10.
        :return: None
        """
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
            raise RuntimeError('No connected network found. Try increasing thresholds.')

        best_config = min(valid, key=lambda x: (x['n_pairs'], x['days'], x['meters']))
        self.baseline_pairs = best_config['df']

    def _compute_interferograms(
        self,
        intensity_wavelength=20,
        phase_wavelength=30,
        coarsen=(1, 4),
        goldstein_patch=8,
    ):
        """Compute interferograms from baseline pairs.

        :param int intensity_wavelength: Gaussian smoothing cut-off wavelength (metres) for intensity. Defaults to 20.
        :param int phase_wavelength: Gaussian smoothing cut-off wavelength (metres) for wrapped phase. Defaults to 30.
        :param tuple coarsen: Radar coordinate downsampling (range_factor, azimuth_factor). Defaults to (1, 4).
        :param int goldstein_patch: Window size (pixels) for Goldstein filtering. Defaults to 8.
        :return: None
        """
        topo = self.sbas.get_topo()
        data = self.sbas.open_data()

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

        self.intf = self.sbas.interferogram(phase_goldstein)

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
        corr_u, intf_u = dask.persist(dec_u(self.corr), dec_u(self.intf))

        # Build Correlation Mask
        corr_mask = corr_u.where(corr_u >= corr_limit)

        # Verify we have valid pixels to unwrap
        n_valid = int(np.isfinite(corr_mask).sum().compute())
        if n_valid == 0:
            raise RuntimeError(
                f'No pixels found above corr_limit={corr_limit}. Unwrapping aborted.'
            )

        # Run SNAPHU Unwrapping
        self.unwrap = self.sbas.unwrap_snaphu(
            intf_u.where(corr_mask), corr_mask
        ).persist()

        # Trigger computation to catch SNAPHU execution errors immediately
        _ = float(self.unwrap.phase.isel(pair=0).mean().compute())

    def _detrend_phase(self, ramp_factor=3.0, base_wavelength=30, chunksize=256):
        """Remove long-wavelength ramps from the unwrapped phase in radar geometry.

        :param float ramp_factor: Multiplier applied to base_wavelength to define the ramp scale. Defaults to 3.0.
        :param int base_wavelength: Reference smoothing wavelength (m). Defaults to 30.
        :param int chunksize: Size of spatial blocks for Dask processing to optimize memory usage. Defaults to 256.
        :return: None
        """
        if self.unwrap is None:
            raise RuntimeError('Phase must be unwrapped before detrending.')

        # Determine spatial scales
        ramp_wavelength = ramp_factor * base_wavelength

        # Ensure Dask chunking
        phase_data = self.unwrap.phase
        chunk_spec = {d: chunksize for d in phase_data.dims if d in ('y', 'x')}
        if chunk_spec:
            phase_data = phase_data.chunk(chunk_spec)

        # Build and subtract the ramp
        ramp = self.sbas.gaussian(phase_data, wavelength=ramp_wavelength).persist()

        # Subtracting the ramp leaves only the localized displacement signal
        self.detrend = (phase_data - ramp).persist()

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
        corr_ra = self.corr
        if self.detrend.sizes != self.corr.sizes:
            corr_ra = self.sbas.decimator(float(target_m))(self.corr).persist()

        with ProgressBar():
            sol = self.sbas.lstsq(self.detrend, corr_ra)
            disp_ra = self.sbas.los_displacement_mm(sol).persist()

        # Reference to Zero (Inline Time-Dim Selection)
        t_dim = next(
            (d for d in disp_ra.dims if d in ('date', 'time', 'epoch', 'pair')), None
        )
        if t_dim:
            disp_ra = disp_ra.where(
                disp_ra[t_dim] != disp_ra[t_dim].values[0], other=np.nan
            )

        # Geocoding & Coordinate Transformation
        self.sbas.compute_geocode(float(target_m))
        self.disp_ll = self.sbas.cropna(
            self.sbas.ra2ll(disp_ra)
        ).compute()  # can be switched to .persist()

        # RMSE Calculation
        disp_pairs_ra = self.sbas.los_displacement_mm(self.detrend).persist()
        self.rmse = self.sbas.rmse(disp_pairs_ra, disp_ra, corr_ra).persist()

    def _compute_risk(self):
        """Compute risk map based on displacements, velocity and slope.

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
        vel_map = np.abs(self.sbas.velocity(disp))  # TODO: Do we want it as an output?

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
        # Temporal & Spatial Setup
        stack_df = self.sbas.to_dataframe()
        dt_index = pd.to_datetime(stack_df.index)
        start_date, stop_date = dt_index.min().date(), dt_index.max().date()

        aoi = loads(aoi) if isinstance(aoi, str) else aoi
        bounds = aoi.bounds
        center_lon, center_lat = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=center_lon, lat=center_lat) or 'UTC'

        # API Client Setup
        cache_session = requests_cache.CachedSession(str(output_dir / 'openmeteo_cache'), expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
        om_client = openmeteo_requests.Client(session=retry_session)

        # Build Grid & Sample Altitudes
        lats, lons = np.linspace(bounds[1], bounds[3], grid_rows), np.linspace(bounds[0], bounds[2], grid_cols)
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
                except:
                    locations[k]['alt'] = 0
        except Exception as e:
            print(f'[ENVDB] Altitude sampling skipped: {e}')

        # Fetch & Process Climate Data
        climate_url = 'https://archive-api.open-meteo.com/v1/archive'
        climate_vars = [
            'temperature_2m_max', 'precipitation_sum',
            'et0_fao_evapotranspiration', 'wind_gusts_10m_max'
        ]
        all_climate = []

        for name, loc in locations.items():
            params = {
                'latitude': loc['lat'], 'longitude': loc['lon'],
                'start_date': start_date.isoformat(), 'end_date': stop_date.isoformat(),
                'daily': climate_vars, 'timezone': tz_name, 'models': model,
            }
            res = om_client.weather_api(climate_url, params=params)[0]
            daily = res.Daily()

            num_days = len(daily.Variables(0).ValuesAsNumpy())
            dates = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit='s', utc=True).tz_convert(tz_name).tz_localize(None),
                periods=num_days, freq='D'
            )

            df = pd.DataFrame({'date': dates, 'location': name})
            for idx, var in enumerate(climate_vars):
                df[var] = daily.Variables(idx).ValuesAsNumpy()

            # Rainfall indicators
            df['precip_7d_mm'] = df['precipitation_sum'].rolling(7, min_periods=1).sum()
            df['precip_30d_mm'] = df['precipitation_sum'].rolling(30, min_periods=1).sum()
            df['wet_days_30d'] = (df['precipitation_sum'] > 1.0).rolling(30, min_periods=1).sum()

            # Hydrological balance (Precip - Evapotranspiration)
            df['daily_wb'] = df['precipitation_sum'] - df['et0_fao_evapotranspiration']
            df['water_balance_30d_mm'] = df['daily_wb'].rolling(30, min_periods=1).sum()

            # Wind Gusts
            df['wind_gust_kmh'] = df['wind_gusts_10m_max']

            # Temperature Anomaly (Current Max vs 30-day average)
            df['temp_max_C_month_anom'] = df['temperature_2m_max'] - df['temperature_2m_max'].rolling(30,
                                                                                                      min_periods=1).mean()

            all_climate.append(df)

        climate_df = pd.concat(all_climate).reset_index(drop=True)

        # Fetch & Process Air Quality Data
        aq_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        aq_vars = ['pm10', 'pm2_5']
        all_aq = []

        # Helper: US-AQI calculation for PM2.5
        def calculate_aqi(pm25):
            if pm25 < 0: return 0
            if pm25 <= 12.0: return ((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0
            if pm25 <= 35.4: return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
            if pm25 <= 55.4: return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
            return 200  # Cap for simplified model

        for name, loc in locations.items():
            params = {
                'latitude': loc['lat'], 'longitude': loc['lon'],
                'start_date': start_date.isoformat(), 'end_date': stop_date.isoformat(),
                'hourly': aq_vars, 'timezone': tz_name,
            }
            res = om_client.weather_api(aq_url, params=params)[0]
            hourly = res.Hourly()

            dates = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit='s', utc=True).tz_convert(tz_name).tz_localize(None),
                periods=len(hourly.Variables(0).ValuesAsNumpy()), freq='h'
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
        climate_df.to_parquet(output_dir / 'climate_daily_db.parquet', index=False)
        air_df.to_parquet(output_dir / 'air_quality_daily_db.parquet', index=False)

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

        :param Path output_dir: Directory path where final results will be saved and 'climate_daily_db.parquet'
                                and 'air_quality_daily_db.parquet' are already stored.
        :return: None
        """
        # Extract and Merge InSAR Layers
        if self.disp_ll is None or self.risk_map is None:
            raise RuntimeError("InSAR displacement or risk map not found. Run previous steps first.")

        # Convert xarray datasets to long-form DataFrames
        df_disp = self.disp_ll.to_dataframe().reset_index()
        df_risk = self.risk_map.to_dataframe().reset_index()

        if "date" in df_risk.columns:
            df_risk = df_risk.drop(columns=["date"])

        # Merge on spatial coordinates (lat, lon)
        df_base = df_disp.merge(df_risk, on=["lat", "lon"])

        # Normalize and Calculate Missing Motion Columns
        rename_map = {
            "displacement": "los_mm",
            "los": "los_mm",
            "slope": "slope_deg",
            "risk_score": "insar_risk_score_base"
        }
        df_base = df_base.rename(columns={k: v for k, v in rename_map.items() if k in df_base.columns})

        # Suffix Fallback: If 'date' was somehow still renamed by pandas
        if "date" not in df_base.columns:
            if "date_x" in df_base.columns:
                df_base = df_base.rename(columns={"date_x": "date"})
            elif "date_y" in df_base.columns:
                df_base = df_base.rename(columns={"date_y": "date"})

        # Ensure cell_id exists for time-series grouping
        df_base["cell_id"] = df_base["lat"].astype(str) + "_" + df_base["lon"].astype(str)
        df_base["date"] = pd.to_datetime(df_base["date"]).dt.normalize()

        # Calculate dlos (daily change) and velocity (rate)
        df_base = df_base.sort_values(["cell_id", "date"])
        df_base["dlos_mm"] = df_base.groupby("cell_id")["los_mm"].diff().fillna(0)

        # Calculate days between acquisitions to get true velocity
        date_diff = df_base.groupby("cell_id")["date"].diff().dt.days.fillna(1)
        date_diff = date_diff.replace(0, 1)  # Prevent division by zero
        df_base["vel_mm_per_day"] = (df_base["dlos_mm"] / date_diff).abs()

        # Load Environmental Data
        clim = pd.read_parquet(output_dir / "climate_daily_db.parquet")
        air = pd.read_parquet(output_dir / "air_quality_daily_db.parquet")

        clim["date"] = pd.to_datetime(clim["date"]).dt.normalize()
        air["date"] = pd.to_datetime(air["date"]).dt.normalize()

        # Risk Thresholds and Weights
        TH = {
            "VEL_DAY_LO": 0.5, "VEL_DAY_HI": 2.0,
            "DLOS_DAY_LO": 0.5, "DLOS_DAY_HI": 2.0,
            "LOS_LO": 20.0, "LOS_HI": 60.0,
            "SLOPE_LO": 10.0, "SLOPE_HI": 20.0,
            "P7_LO": 20.0, "P7_HI": 60.0,
            "P30_LO": 60.0, "P30_HI": 150.0,
            "PM25_LO": 15.0, "PM25_HI": 35.0,
            "RISK_MODERATE": 35.0, "RISK_HIGH": 60.0, "RISK_CRITICAL": 80.0,
        }
        W = {"motion": 0.45, "terrain": 0.15, "hydroclimate": 0.25, "atmosphere": 0.15}

        def _lin_score(x, lo, hi):
            return np.clip((np.asarray(x, dtype="float64") - lo) / (hi - lo + 1e-12), 0.0, 1.0)

        # Spatial Matching (Robust Nearest Neighbor)
        cell_lut = df_base[["cell_id", "lat", "lon"]].drop_duplicates("cell_id")

        # Check if environmental data has coordinates for distance matching
        has_env_coords = "lat" in clim.columns and "lon" in clim.columns

        if has_env_coords:
            env_pts = clim[["location", "lat", "lon"]].drop_duplicates("location")
            p_lat, p_lon = env_pts["lat"].values, env_pts["lon"].values
            c_lat, c_lon = cell_lut["lat"].values, cell_lut["lon"].values

            dist = (c_lat[:, None] - p_lat[None, :]) ** 2 + (c_lon[:, None] - p_lon[None, :]) ** 2
            cell_lut["env_location"] = env_pts["location"].values[dist.argmin(axis=1)]
        else:
            # Fallback: Map to the first location name if coordinates are missing
            primary_loc = clim["location"].iloc[0]
            cell_lut["env_location"] = primary_loc

        # Merge Databases
        df_merged = df_base.merge(cell_lut[["cell_id", "env_location"]], on="cell_id")

        # Prefix columns to avoid collisions
        clim_cols = {c: f"climate_{c}" for c in clim.columns if c not in ["date", "location"]}
        air_cols = {c: f"air_{c}" for c in air.columns if c not in ["date", "location"]}

        df_merged = df_merged.merge(clim.rename(columns=clim_cols),
                                    left_on=["date", "env_location"],
                                    right_on=["date", "location"], how="left")
        df_merged = df_merged.merge(air.rename(columns=air_cols),
                                    left_on=["date", "env_location"],
                                    right_on=["date", "location"], how="left")

        # Compute Composite Risk Score
        # Motion
        v_s = _lin_score(df_merged["vel_mm_per_day"], TH["VEL_DAY_LO"], TH["VEL_DAY_HI"])
        d_s = _lin_score(np.abs(df_merged["dlos_mm"]), TH["DLOS_DAY_LO"], TH["DLOS_DAY_HI"])
        l_s = _lin_score(np.abs(df_merged["los_mm"]), TH["LOS_LO"], TH["LOS_HI"])
        motion_comp = 0.40 * v_s + 0.35 * d_s + 0.25 * l_s

        # Other Components
        hydro_comp = _lin_score(df_merged.get("climate_precip_7d_mm", 0), TH["P7_LO"], TH["P7_HI"])
        terrain_comp = _lin_score(df_merged.get("slope_deg", 0), TH["SLOPE_LO"], TH["SLOPE_HI"])
        atmos_comp = _lin_score(df_merged.get("air_pm2_5", 0), TH["PM25_LO"], TH["PM25_HI"])

        # Final Risk Score (0-100)
        total_risk = 100.0 * (W["motion"] * motion_comp +
                              W["terrain"] * terrain_comp +
                              W["hydroclimate"] * hydro_comp +
                              W["atmosphere"] * atmos_comp)

        df_merged["risk_score_0to100"] = total_risk.astype("float32")

        # Classification
        level = np.full(len(df_merged), "Low", dtype=object)
        level[total_risk >= TH["RISK_MODERATE"]] = "Moderate"
        level[total_risk >= TH["RISK_HIGH"]] = "High"
        level[total_risk >= TH["RISK_CRITICAL"]] = "Critical"
        df_merged["risk_class"] = pd.Categorical(level,
                                                 categories=["Low", "Moderate", "High", "Critical"],
                                                 ordered=True)

        # UTM Projection
        mean_lat, mean_lon = df_merged["lat"].mean(), df_merged["lon"].mean()
        utm_zone = int(np.floor((mean_lon + 180.0) / 6.0) + 1)
        epsg = 32700 + utm_zone if mean_lat < 0 else 32600 + utm_zone

        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xe, yn = transformer.transform(df_merged["lon"].values, df_merged["lat"].values)
        df_merged["UTM_E"], df_merged["UTM_N"] = xe, yn

        # Export
        df_merged.to_parquet(output_dir / "final_risk_database.parquet", index=False)

        # Save Metadata
        governance = {
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "thresholds": TH,
            "weights": W,
            "utm_epsg": int(epsg)
        }
        with open(output_dir / "risk_governance.json", "w") as f:
            json.dump(governance, f, indent=2)

    def _run(self):
        self._download_orbits(self._config['datadir'])
        self._download_dem(self._config['aoi'], self._config['dem_path'])
        self._download_landmask(self._config['aoi'], self._config['landmask_path'])
        self._run_dask_cluster()  # TODO: how to handle **kwargs?
        self._stack_scenes(self._config['datadir'], self._config['workdir'])
        self._reframe_scenes(self._config['aoi'])
        self._load_dem_and_landmask(
            self._config['aoi'], self._config['dem_path'], self._config['landmask_path']
        )
        self._align_images()
        self._geocoding_transform()
        self._find_optimal_network()
        self._compute_interferograms()
        self._unwrap_phase()
        self._detrend_phase()
        self._compute_displacement()
        self._compute_risk()
        self._environmental_database(self._config['aoi'], self._config['result_dir'])
        self._compute_risk_database(self._config['result_dir'])
        pass
