from insardev_pygmtsar import S1
from insardev_toolkit import EOF, Tiles
import xarray as xr
import rioxarray
import numpy as np

from .base import BasePipeline


class Sentinel1Pipeline(BasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dem_da = None
        self.dem_masked = None
        self.dem_cropped = None

    def _download_orbits(self, datadir):
        s1 = S1(datadir)
        EOF().download(datadir, s1.to_dataframe())

    def _download_dem_baseline(self, roi_square):
        tiles = Tiles()
        dem_raw = tiles.download_dem(roi_square, skip_exist=False)

        if isinstance(dem_raw, xr.Dataset):
            candidates = [v for v in dem_raw.data_vars if dem_raw[v].ndim >= 2]
            if not candidates:
                raise RuntimeError("DEM Dataset has no 2D variables.")
            da = dem_raw[candidates[0]]
        elif isinstance(dem_raw, xr.DataArray):
            da = dem_raw
        else:
            raise TypeError(f"Unsupported DEM object type: {type(dem_raw)}")

        da = da.squeeze(drop=True)
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)

        da = da.squeeze(drop=True)
        if da.ndim != 2:
            raise RuntimeError(f"Copernicus baseline: Expected 2D after squeeze, got dims={da.dims}, shape={da.shape}")
        if "lat" in da.dims and "lon" in da.dims:
            da = da.transpose("lat", "lon")
        elif "y" in da.dims and "x" in da.dims:
            da = da.rename({"y": "lat", "x": "lon"})
            da = da.transpose("lat", "lon")
        elif "latitude" in da.dims and "longitude" in da.dims:
            da = da.rename({"latitude": "lat", "longitude": "lon"})
            da = da.transpose("lat", "lon")
        else:
            raise RuntimeError(
                f"Copernicus baseline: Cannot normalise to (lat, lon). dims={da.dims}, shape={da.shape}"
            )

        self.dem_da = da.rio.write_crs("EPSG:4326")
        self.dem_da = self.dem_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

    def _lidar_infill(self, lidar_data):
        use_lidar = lidar_data.is_file()

        if use_lidar:
            print(f"[DEM] LiDAR DEM found at {lidar_data}. Merging LiDAR over Copernicus where finite...")

            ds_lidar = xr.open_dataset(lidar_data)
            print("[DEM] LiDAR variables:", list(ds_lidar.data_vars))

            # Prefer variable named 'z', else first 2D var
            if "z" in ds_lidar.data_vars and ds_lidar["z"].ndim >= 2:
                dem_lidar = ds_lidar["z"]
            else:
                cand = [v for v in ds_lidar.data_vars if ds_lidar[v].ndim >= 2]
                if not cand:
                    ds_lidar.close()
                    raise RuntimeError(f"LiDAR NetCDF has no 2D variables: {list(ds_lidar.data_vars)}")
                dem_lidar = ds_lidar[cand[0]]

            dem_lidar = dem_lidar.squeeze(drop=True)
            if "band" in dem_lidar.dims:
                dem_lidar = dem_lidar.isel(band=0, drop=True)

            if dem_lidar.ndim != 2:
                ds_lidar.close()
                raise RuntimeError(f"LiDAR DEM is not 2D: dims={dem_lidar.dims}, shape={dem_lidar.shape}")

            # Normalise LiDAR dims to lat/lon where possible
            if "lat" in dem_lidar.dims and "lon" in dem_lidar.dims:
                dem_lidar_rio = dem_lidar.transpose("lat", "lon")
                dem_lidar_rio = dem_lidar_rio.rio.write_crs("EPSG:4326")
                dem_lidar_rio = dem_lidar_rio.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            elif "latitude" in dem_lidar.dims and "longitude" in dem_lidar.dims:
                dem_lidar_rio = dem_lidar.rename({"latitude": "lat", "longitude": "lon"}).transpose("lat", "lon")
                dem_lidar_rio = dem_lidar_rio.rio.write_crs("EPSG:4326")
                dem_lidar_rio = dem_lidar_rio.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            else:
                # Fallback: assume 2D dims are y/x-like and mark CRS (best-effort)
                y_dim, x_dim = dem_lidar.dims
                dem_lidar_rio = dem_lidar.rio.write_crs("EPSG:4326")
                dem_lidar_rio = dem_lidar_rio.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)

            # Reproject LiDAR to match Copernicus grid
            dem_lidar_on_cop = dem_lidar_rio.rio.reproject_match(self.dem_da)

            # Force consistent dims/coords after reproject (avoids odd naming drift)
            dem_lidar_on_cop = xr.DataArray(
                data=dem_lidar_on_cop.values,
                coords={"lat": self.dem_da["lat"].values, "lon": self.dem_da["lon"].values},
                dims=("lat", "lon"),
                name="dem_lidar_on_cop",
            )

            # LiDAR overrides Copernicus where LiDAR is finite
            self.dem_da = xr.where(np.isfinite(dem_lidar_on_cop), dem_lidar_on_cop, self.dem_da)

            ds_lidar.close()

            # Restore rioxarray metadata after xr.where
            self.dem_da = self.dem_da.rio.write_crs("EPSG:4326")
            self.dem_da = self.dem_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

            print("[DEM] LiDAR merge complete. Composite DEM shape:", self.dem_da.shape)
        else:
            print(f"[DEM] LiDAR DEM not found at {lidar_data}. Using Copernicus DEM only.")

    def _save_composite_dem(self, output_file):
        self.dem_da = self.dem_da.load().squeeze(drop=True)
        self.dem_da.name = "dem"
        self.dem_da = self.dem_da.rio.write_crs("EPSG:4326")
        self.dem_da = self.dem_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

        xr.Dataset({"dem": self.dem_da}).to_netcdf(output_file)
        print(f"[DEM] Composite DEM written to: {output_file}")

        with xr.open_dataset(output_file) as ds_check:
            if "dem" not in ds_check.data_vars:
                raise RuntimeError(
                    f"[DEM] Saved file does not contain variable 'dem'. Found: {list(ds_check.data_vars)}")
            print("[DEM] Saved variables:", list(ds_check.data_vars))

    def _clip_dem(self, aoi):
        # TODO: Check if really needed
        self.dem_masked = self.dem_da.rio.clip([aoi], self.dem_da.rio.crs, drop=False, invert=False)
        self.dem_cropped = self.dem_da.rio.clip([aoi], self.dem_da.rio.crs, drop=True, invert=False)

        print("Original DEM shape:", self.dem_da.shape)
        print("Masked DEM shape  :", self.dem_masked.shape, "(same as original)")
        print("Cropped DEM shape :", self.dem_cropped.shape, "(smaller)")

    def run(self, data_dir, bbox, lidar_file, output_dem, aoi):
        self._download_orbits(data_dir)
        self._download_dem_baseline(bbox)
        self._lidar_infill(lidar_file)
        self._save_composite_dem(output_dem)
        self._clip_dem(aoi)
        # ...
