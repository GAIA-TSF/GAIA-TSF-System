from insardev_pygmtsar import S1
from insardev_toolkit import ASF, EOF, Tiles
import xarray as xr
import rioxarray # noqa: F401
import numpy as np
import pandas as pd
from pathlib import Path

from .base import BasePipeline


class Sentinel1Pipeline(BasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bursts = None
        self.s1 = None
        self.dem_da = None
        self.dem_masked = None
        self.dem_cropped = None
        self.landmask_arr = None
        self.ref_date = None

    def _search_bursts(self, aoi, start, end, direction):
        """Search Sentinel-1 BURST data intersecting with chosen AOI using ASF.

        :param str aoi: WKT string representing the area of interest.
        :param str start: Start time for search in 'yyyy-mm-dd' format.
        :param str end: End time for search in 'yyyy-mm-dd' format.
        :param str direction: Flight direction of Sentinel-1 satellites ('A' - ascending, 'D' - descending).
        :return: None
        """
        self.bursts = ASF.search(
            aoi, startTime=start, stopTime=end, flightDirection=direction
        )

    def _download_bursts(self, username, password, datadir):
        """Download searched Sentinel-1 BURST data using ASF.

        :param str username: Username for ASF.
        :param str password: Password for ASF.
        :param str datadir: Directory path to save the data.
        :return: None
        """
        burst_ids = self.bursts.fileID.tolist()
        asf = ASF(username, password)
        asf.download(datadir, burst_ids)

    def _download_orbits(self, datadir):
        """Download precise orbit files for Sentinel-1 BURST data.

        :param str datadir: Path to directory with downloaded BURST data
        :return: None.
        """
        self.s1 = S1(datadir)
        EOF().download(datadir, self.s1.to_dataframe())

    def _download_dem_baseline(self, roi_square):
        """Download DEM baseline for AOI.

        :param Polygon roi_square: Bounding box of the area of interest as a Shapely Polygon.
        :return: None
        """
        tiles = Tiles()
        dem_raw = tiles.download_dem(roi_square, skip_exist=False)

        if isinstance(dem_raw, xr.Dataset):
            candidates = [v for v in dem_raw.data_vars if dem_raw[v].ndim >= 2]
            if not candidates:
                raise RuntimeError('DEM Dataset has no 2D variables.')
            da = dem_raw[candidates[0]]
        elif isinstance(dem_raw, xr.DataArray):
            da = dem_raw
        else:
            raise TypeError(f'Unsupported DEM object type: {type(dem_raw)}')

        da = da.squeeze(drop=True)
        if 'band' in da.dims:
            da = da.isel(band=0, drop=True)

        da = da.squeeze(drop=True)
        if da.ndim != 2:
            raise RuntimeError(
                f'Copernicus baseline: Expected 2D after squeeze, got dims={da.dims}, shape={da.shape}'
            )
        if 'lat' in da.dims and 'lon' in da.dims:
            da = da.transpose('lat', 'lon')
        elif 'y' in da.dims and 'x' in da.dims:
            da = da.rename({'y': 'lat', 'x': 'lon'})
            da = da.transpose('lat', 'lon')
        elif 'latitude' in da.dims and 'longitude' in da.dims:
            da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
            da = da.transpose('lat', 'lon')
        else:
            raise RuntimeError(
                f'Copernicus baseline: Cannot normalise to (lat, lon). dims={da.dims}, shape={da.shape}'
            )

        self.dem_da = da.rio.write_crs('EPSG:4326')
        self.dem_da = self.dem_da.rio.set_spatial_dims(
            x_dim='lon', y_dim='lat', inplace=False
        )

    def _lidar_infill(self, lidar_data):
        """Infill DEM with more precise LiDAR data.

        :param Path lidar_data: Path to LiDAR data.
        :return: None
        """
        use_lidar = lidar_data.is_file()

        if use_lidar:
            print(
                f'[DEM] LiDAR DEM found at {lidar_data}. Merging LiDAR over Copernicus where finite...'
            )

            ds_lidar = xr.open_dataset(lidar_data)
            print('[DEM] LiDAR variables:', list(ds_lidar.data_vars))

            # Prefer variable named 'z', else first 2D var
            if 'z' in ds_lidar.data_vars and ds_lidar['z'].ndim >= 2:
                dem_lidar = ds_lidar['z']
            else:
                cand = [v for v in ds_lidar.data_vars if ds_lidar[v].ndim >= 2]
                if not cand:
                    ds_lidar.close()
                    raise RuntimeError(
                        f'LiDAR NetCDF has no 2D variables: {list(ds_lidar.data_vars)}'
                    )
                dem_lidar = ds_lidar[cand[0]]

            dem_lidar = dem_lidar.squeeze(drop=True)
            if 'band' in dem_lidar.dims:
                dem_lidar = dem_lidar.isel(band=0, drop=True)

            if dem_lidar.ndim != 2:
                ds_lidar.close()
                raise RuntimeError(
                    f'LiDAR DEM is not 2D: dims={dem_lidar.dims}, shape={dem_lidar.shape}'
                )

            # Normalise LiDAR dims to lat/lon where possible
            if 'lat' in dem_lidar.dims and 'lon' in dem_lidar.dims:
                dem_lidar_rio = dem_lidar.transpose('lat', 'lon')
                dem_lidar_rio = dem_lidar_rio.rio.write_crs('EPSG:4326')
                dem_lidar_rio = dem_lidar_rio.rio.set_spatial_dims(
                    x_dim='lon', y_dim='lat', inplace=False
                )
            elif 'latitude' in dem_lidar.dims and 'longitude' in dem_lidar.dims:
                dem_lidar_rio = dem_lidar.rename(
                    {'latitude': 'lat', 'longitude': 'lon'}
                ).transpose('lat', 'lon')
                dem_lidar_rio = dem_lidar_rio.rio.write_crs('EPSG:4326')
                dem_lidar_rio = dem_lidar_rio.rio.set_spatial_dims(
                    x_dim='lon', y_dim='lat', inplace=False
                )
            else:
                # Fallback: assume 2D dims are y/x-like and mark CRS (best-effort)
                y_dim, x_dim = dem_lidar.dims
                dem_lidar_rio = dem_lidar.rio.write_crs('EPSG:4326')
                dem_lidar_rio = dem_lidar_rio.rio.set_spatial_dims(
                    x_dim=x_dim, y_dim=y_dim, inplace=False
                )

            # Reproject LiDAR to match Copernicus grid
            dem_lidar_on_cop = dem_lidar_rio.rio.reproject_match(self.dem_da)

            # Force consistent dims/coords after reproject (avoids odd naming drift)
            dem_lidar_on_cop = xr.DataArray(
                data=dem_lidar_on_cop.values,
                coords={
                    'lat': self.dem_da['lat'].values,
                    'lon': self.dem_da['lon'].values,
                },
                dims=('lat', 'lon'),
                name='dem_lidar_on_cop',
            )

            # LiDAR overrides Copernicus where LiDAR is finite
            self.dem_da = xr.where(
                np.isfinite(dem_lidar_on_cop), dem_lidar_on_cop, self.dem_da
            )

            ds_lidar.close()

            # Restore rioxarray metadata after xr.where
            self.dem_da = self.dem_da.rio.write_crs('EPSG:4326')
            self.dem_da = self.dem_da.rio.set_spatial_dims(
                x_dim='lon', y_dim='lat', inplace=False
            )

            print('[DEM] LiDAR merge complete. Composite DEM shape:', self.dem_da.shape)
        else:
            print(
                f'[DEM] LiDAR DEM not found at {lidar_data}. Using Copernicus DEM only.'
            )

    def _save_composite_dem(self, output_file):
        """Save a composite DEM to disk.

        :param Path output_file: Path to output file.
        :return: None
        """
        self.dem_da = self.dem_da.load().squeeze(drop=True)
        self.dem_da.name = 'dem'
        self.dem_da = self.dem_da.rio.write_crs('EPSG:4326')
        self.dem_da = self.dem_da.rio.set_spatial_dims(
            x_dim='lon', y_dim='lat', inplace=False
        )

        xr.Dataset({'dem': self.dem_da}).to_netcdf(output_file)
        print(f'[DEM] Composite DEM written to: {output_file}')

        with xr.open_dataset(output_file) as ds_check:
            if 'dem' not in ds_check.data_vars:
                raise RuntimeError(
                    f"[DEM] Saved file does not contain variable 'dem'. Found: {list(ds_check.data_vars)}"
                )
            print('[DEM] Saved variables:', list(ds_check.data_vars))

    def _clip_dem(self, aoi):
        """Clip DEM to AOI.

        :param str aoi: WKT string representing the area of interest.
        :return: None
        """
        # TODO: Check if really needed
        self.dem_masked = self.dem_da.rio.clip(
            [aoi], self.dem_da.rio.crs, drop=False, invert=False
        )
        self.dem_cropped = self.dem_da.rio.clip(
            [aoi], self.dem_da.rio.crs, drop=True, invert=False
        )

    def _save_landmask(self, output_landmask):
        """Save landmask based on DEM to disk.

        :param Path output_landmask: Path to output landmask file.
        :return: None
        """
        # TODO: Check if really needed
        self.landmask_arr = xr.where(np.isfinite(self.dem_masked), 1, 0).astype('uint8')
        self.landmask_arr.name = 'landmask'
        self.landmask_arr = self.landmask_arr.rio.write_crs(self.dem_da.rio.crs)
        self.landmask_arr = self.landmask_arr.rio.set_spatial_dims(
            x_dim='lon', y_dim='lat', inplace=False
        )

        xr.Dataset({'landmask': self.landmask_arr}).to_netcdf(output_landmask)

        print(f'[LANDMASK] AOI-based landmask saved to: {output_landmask}')
        print(
            '[LANDMASK] Convention: 1 = inside AOI (TSF polygon), 0 = outside within DEM area.'
        )
        print(
            '[LANDMASK] Landmask dims:', self.landmask_arr.dims, self.landmask_arr.shape
        )

    def _link_s1_with_dem(self, datadir, dem_file):
        """Link Sentinel-1 BURST data with DEM.

        :param str datadir: Path to directory with BURST data.
        """
        self.s1 = S1(datadir, DEM=str(dem_file))
        self.s1.to_dataframe()

    def _infer_ref_date(self):
        """Find the reference date from Sentinel-1 BURST data.

        :return: None
        """
        if self.bursts is not None and len(self.bursts) > 0:
            acq_dates = pd.to_datetime(self.bursts['startTime'], utc=True)
            self.ref_date = acq_dates.iloc[
                (acq_dates - acq_dates.median()).abs().idxmin()
            ].strftime('%Y-%m-%d')
        elif self.s1 is not None:
            acq_dates = pd.to_datetime(self.s1.df['startTime'], utc=True)
            self.ref_date = acq_dates.loc[
                (acq_dates - acq_dates.median()).abs().idxmin()
            ].strftime('%Y-%m-%d')
        else:
            raise NameError(
                'No data found to infer reference date. Need one of: bursts, or s1.'
            )

    def _transform_to_zarr(self, dem_file, datadir, zarrdir):
        """Transform Sentinel-1 BURST data to georeferenced zarr format. GMTSAR tool needs to be installed!
        See: https://github.com/gmtsar/gmtsar/wiki

        :param Path dem_file: Path to Sentinel-1 BURST data.
        :param Path|str datadir: Path to directory with BURST data.
        :param Path zarrdir: Path to output zarr directory.
        :return: None
        """
        dem_path = Path(dem_file)
        if dem_path.is_file():
            ds = xr.open_dataset(str(dem_path))
            da = ds["dem"]
            da = da.copy()
            da.name = "dem"
            ds.close()

            dem_onevar = dem_path.with_name(dem_path.stem + "_onevar.nc")
            da.to_netcdf(str(dem_onevar))

            if datadir.is_dir() and zarrdir.is_dir():
                if self.ref_date is not None:
                    self.s1 = S1(datadir, DEM=str(dem_onevar))
                    self.s1.transform(zarrdir, ref=self.ref_date, n_jobs=1)
                else:
                    raise NameError('No reference date found. Run _infer_ref_date() first.')
            else:
                raise NameError('datadir or zarrdir or both do not exist.')
        else:
            raise FileNotFoundError(dem_path)

    def run(
        self,
        aoi,
        start,
        end,
        direction,
        username,
        password,
        data_dir,
        bbox,
        lidar_file,
        output_dem,
        output_landmask,
        zarrdir,
    ):
        self._search_bursts(aoi, start, end, direction)
        self._download_bursts(username, password, data_dir)
        self._download_orbits(data_dir)
        self._download_dem_baseline(bbox)
        self._lidar_infill(lidar_file)
        self._save_composite_dem(output_dem)
        self._clip_dem(aoi)
        self._save_landmask(output_landmask)
        self._link_s1_with_dem(data_dir, output_dem)
        self._infer_ref_date()
        self._transform_to_zarr(output_dem, data_dir, zarrdir)
        # ...
