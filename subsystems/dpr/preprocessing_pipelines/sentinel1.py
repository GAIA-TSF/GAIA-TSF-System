from .base import BasePipeline

from pathlib import Path
from pygmtsar import ASF, S1, Tiles, Stack
from dask.distributed import Client

class Sentinel1Pipeline(BasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bursts = None
        self.client = None
        self.s1 = None
        self.stack = None
        self.dem = None
        self.landmask = None
        self.sbas = None
        self.dem_masked = None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _search_bursts(self, aoi, start, end, direction):
        """Search Sentinel-1 BURST data intersecting with chosen AOI using ASF.

        :param str aoi: WKT string representing the area of interest.
        :param str start: Start time for search in 'yyyy-mm-dd' format.
        :param str end: End time for search in 'yyyy-mm-dd' format.
        :param str direction: Flight direction of Sentinel-1 satellites ('A' - ascending, 'D' - descending).
        :return: None
        """
        all_bursts = ASF.search(
            aoi, startTime=start, stopTime=end, flightDirection=direction
        )
        best_orbit = all_bursts['pathNumber'].value_counts().idxmax()
        self.bursts = all_bursts[all_bursts['pathNumber'] == best_orbit]

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
            Tiles().download_landmask(aoi, filename=output_landmask, skip_exist=True).fillna(0)
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
                f"Safety Triggered: datadir and workdir are the same location ({data_path}). "
                "Aborting to prevent accidental data deletion."
            )
        if work_path in data_path.parents:
            raise ValueError(
                f"Safety Triggered: workdir ({work_path}) is a parent of datadir ({data_path}). "
                "Aborting to prevent accidental data deletion."
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
        self.dem_masked = self.sbas.get_dem().where(self.sbas.get_landmask()) # maybe not needed

    def _align_images(self):
        """Align Sentinel-1 images.

        :return: None
        """
        self.sbas.compute_align()

    def _geocoding_transform(self):
        """Geocode Sentinel-1 images.

        :return: None
        """
        self.sbas.compute_geocode(coarsen=10.)

    def run(self):
        pass
