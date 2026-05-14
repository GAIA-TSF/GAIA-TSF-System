from __future__ import annotations

from geopandas import GeoDataFrame
from shapely.wkt import loads
from shapely.geometry.base import BaseGeometry
from pygmtsar import ASF
from subsystems.eou.data_acquisition_gateway.base_backend import DataAcquisitionBackend


class ASFDataAcquisitionBackend(DataAcquisitionBackend):
    def search(
        self,
        aoi: str | BaseGeometry,
        start: str,
        end: str,
        direction: str,
        path_number: int | None = None,
        **kwargs,
    ) -> GeoDataFrame:
        """Search for Sentinel-1 BURST data using ASF backend.

        :param str | BaseGeometry aoi: geometry as WKT or shapely BaseGeometry object
        :param str start: start date to be used for temporal filter
        :param str end: end date to be used for temporal filter
        :param str direction: flight direction of Sentinel-1 satellites ('A' - ascending, 'D' - descending).
        :param int path_number: orbit path number to be searched for (if None, the most common path number is selected)
        :return: a collection of BURST data matching the criteria
        :rtype: GeoDataFrame
        """
        if isinstance(aoi, str):
            try:
                aoi = loads(aoi)
            except Exception as e:
                raise ValueError(f'Failed to parse AOI WKT string: {e}')

        all_results = ASF.search(
            aoi, startTime=start, stopTime=end, flightDirection=direction, **kwargs
        )
        if path_number is None:
            best_orbit = all_results['pathNumber'].value_counts().idxmax()
            return all_results[all_results['pathNumber'] == best_orbit]
        else:
            return all_results[all_results['pathNumber'] == path_number]

    def _download(self, search_results: GeoDataFrame, target_dir: str, **kwargs) -> str:
        """Download selected Sentinel-1 BURST data using ASF backend.

        :param GeoDataFrame search_results: search results to be downloaded returned by search method
        :param str target_dir: target directory to store downloaded product
        :return: a path to the directory with downloaded data
        :rtype: str
        """
        username = self.config.get('auth', {}).get('credentials', {}).get('username')
        password = self.config.get('auth', {}).get('credentials', {}).get('password')
        asf = ASF(username, password)
        asf.download(target_dir, search_results.fileID.tolist(), **kwargs)

        return target_dir

    def _download_all(self, search_results: GeoDataFrame, target_dir: str, **kwargs) -> str:
        """Download all selected Sentinel-1 BURST data using ASF backend.
        This acts as a wrapper that redirects to the standard _download method.

        :param GeoDataFrame search_results: search results to be downloaded returned by search method
        :param str target_dir: target directory to store downloaded product
        :return: a path to the directory with downloaded data
        :rtype: str
        """
        return self._download(search_results, target_dir, **kwargs)
