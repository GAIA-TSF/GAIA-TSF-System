from __future__ import annotations

from geopandas import GeoDataFrame
from insardev_toolkit import ASF
from subsystems.eou.data_acquisition_gateway.base_backend import DataAcquisitionBackend


class ASFDataAcquisitionBackend(DataAcquisitionBackend):
    def __init__(self):
        self.config = {}

    def set_config(self, config: dict) -> None:
        self.config = config

    def search(self, aoi: str, start: str, end: str, direction: str, **kwargs):
        """Search Sentinel-1 BURST data using ASF."""
        results = ASF.search(
            aoi,
            startTime=start,
            stopTime=end,
            flightDirection=direction,
            **kwargs
        )
        return results

    def download(self, datadir: str, search_results: GeoDataFrame, **kwargs) -> None:
        """Download Sentinel-1 BURST data using ASF."""
        username = self.config.get('username')
        password = self.config.get('password')
        asf = ASF(username, password)
        asf.download(datadir, search_results.fileID.tolist(), **kwargs)
