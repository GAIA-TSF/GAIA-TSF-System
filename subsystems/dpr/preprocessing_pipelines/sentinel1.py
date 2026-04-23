from .base import BasePipeline

from pathlib import Path
from pygmtsar import S1, Tiles, Stack
from dask.distributed import Client

class Sentinel1Pipeline(BasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None
        self.s1 = None
        self.dem = None
        self.landmask = None
        self.sbas = None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _download_orbits(self, datadir):
        self.s1 = S1.scan_slc(datadir)
        S1.download_orbits(datadir, self.s1)

    def _download_dem(self, aoi, output_dem):
        Tiles().download_dem(aoi, filename=output_dem, skip_exist=False)
        self.dem = output_dem

    def _download_landmask(self, aoi, output_landmask):
        Tiles().download_landmask(aoi, filename=output_landmask).fillna(0)
        self.landmask = output_landmask

    def _run_dask_cluster(self, **kwargs):
        if self.client is None:
            self.client = Client(**kwargs)

    def _stack_scenes(self, datadir, workdir):
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

    def run(self):
        pass
