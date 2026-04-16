from __future__ import annotations

from geopandas import GeoDataFrame
from shapely.wkt import loads
from shapely.geometry.base import BaseGeometry
from insardev_toolkit import ASF
from subsystems.eou.data_acquisition_gateway.base_backend import DataAcquisitionBackend


class ASFDataAcquisitionBackend(DataAcquisitionBackend):
    def search(
        self, aoi: str | BaseGeometry, start: str, end: str, direction: str, **kwargs
    ) -> GeoDataFrame:
        """Search for Sentinel-1 BURST data using ASF backend.

        :param str | BaseGeometry aoi: geometry as WKT or shapely BaseGeometry object
        :param str start: start date to be used for temporal filter
        :param str end: end date to be used for temporal filter
        :param str direction: flight direction of Sentinel-1 satellites ('A' - ascending, 'D' - descending).
        :return: a collection of BURST data matching the criteria
        :rtype: GeoDataFrame
        """
        if isinstance(aoi, str):
            try:
                aoi = loads(aoi)
            except Exception as e:
                raise ValueError(f'Failed to parse AOI WKT string: {e}')

        results = ASF.search(
            aoi, startTime=start, stopTime=end, flightDirection=direction, **kwargs
        )
        return results

    def download(self, search_results: GeoDataFrame, **kwargs) -> str:
        """Download selected Sentinel-1 BURST data using ASF backend.

        :param GeoDataFrame search_results: search results to be downloaded returned by search method
        :return: a path to the directory with downloaded data
        :rtype: str
        """
        username = self.config.get('auth', {}).get('credentials', {}).get('username')
        password = self.config.get('auth', {}).get('credentials', {}).get('password')
        asf = ASF(username, password)
        asf.download(self.output_dir, search_results.fileID.tolist(), **kwargs)

        return self.output_dir
