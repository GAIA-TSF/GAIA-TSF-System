from insardev_pygmtsar import S1
from insardev_toolkit import EOF, Tiles
import xarray as xr
import rioxarray

from .base import BasePipeline


class Sentinel1Pipeline(BasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
    }

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
            da.transpose("lat", "lon")
        elif "y" in da.dims and "x" in da.dims:
            da = da.rename({"y": "lat", "x": "lon"})
            da.transpose("lat", "lon")
        elif "latitude" in da.dims and "longitude" in da.dims:
            da = da.rename({"latitude": "lat", "longitude": "lon"})
            da.transpose("lat", "lon")
        else:
            raise RuntimeError(
                f"Copernicus baseline: Cannot normalise to (lat, lon). dims={da.dims}, shape={da.shape}"
            )

        dem_da = da.rio.write_crs("EPSG:4326")
        dem_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

        return dem_da

    def _build_sbas_stack(self):
        raise NotImplementedError()

    def _reframe_sbas(self):
        raise NotImplementedError()

    def run(self):
        self._build_sbas_stack()
        self._reframe_sbas
        # ...
