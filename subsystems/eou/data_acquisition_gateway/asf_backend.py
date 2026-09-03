from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, List

import re
from pathlib import Path

from geopandas import GeoDataFrame
from shapely.wkt import loads
from shapely.geometry.base import BaseGeometry
from pygmtsar import ASF

from subsystems.eou.data_acquisition_gateway.base_backend import DataAcquisitionBackend
from lib.exceptions import GaiaDataError


class ASFDataAcquisitionBackend(DataAcquisitionBackend):
    def _search(
        self,
        geom: str | BaseGeometry,
        start: str,
        end: str,
        direction: str,
        path_number: int | None = None,
        **kwargs,
    ) -> GeoDataFrame:
        """Search for Sentinel-1 BURST data using ASF backend.

        :param str | BaseGeometry geom: geometry as WKT or shapely BaseGeometry object
        :param str start: start date to be used for temporal filter
        :param str end: end date to be used for temporal filter
        :param str direction: flight direction of Sentinel-1 satellites ('A' - ascending, 'D' - descending).
        :param int path_number: orbit path number to be searched for (if None, the most common path number is selected)
        :return: a collection of BURST data matching the criteria
        :rtype: GeoDataFrame
        """
        if isinstance(geom, str):
            try:
                geom = loads(geom)
            except Exception as e:
                raise GaiaDataError(f'Failed to parse AOI WKT string: {e}')

        all_results = ASF.search(
            geom, startTime=start, stopTime=end, flightDirection=direction, **kwargs
        )
        if path_number is None:
            best_orbit = all_results['pathNumber'].value_counts().idxmax()
            return all_results[all_results['pathNumber'] == best_orbit]
        else:
            return all_results[all_results['pathNumber'] == path_number]

    def _download(
        self, search_results: GeoDataFrame, target_dir: str, **kwargs
    ) -> Path:
        """Download selected Sentinel-1 BURST data using ASF backend.

        :param GeoDataFrame search_results: search results to be downloaded returned by search method
        :param str target_dir: target directory to store downloaded product
        :return: a path to the directory with downloaded data
        :rtype: str
        """
        return self._download_all(search_results, target_dir, **kwargs)[0]

    def _download_all(
        self, search_results: GeoDataFrame, target_dir: str, **kwargs
    ) -> List[Path]:
        """Download all selected Sentinel-1 BURST data using ASF backend.
        This acts as a wrapper that redirects to the standard _download method.

        :param GeoDataFrame search_results: search results to be downloaded returned by search method
        :param str target_dir: target directory to store downloaded product
        :return: a path to the download data
        :rtype: List[Path]
        """
        username = self.config.get('auth', {}).get('credentials', {}).get('username')
        password = self.config.get('auth', {}).get('credentials', {}).get('password')
        asf = ASF(username, password)
        asf.download(target_dir, search_results.fileID.tolist(), **kwargs)

        return self.__resolve_safe_paths(search_results.fileID.tolist(), target_dir)

    def __resolve_safe_paths(
        self,
        file_ids: Iterable[str],
        target_dir: str | Path,
    ) -> list[Path]:
        """
        Resolve SAFE directory paths for given ASF file IDs.

        Matches each ``file_id`` (e.g. ``*-BURST``) to a corresponding
        ``.SAFE`` directory in ``target_dir`` using the common suffix
        identifier.

        :param file_ids: Iterable of ASF file IDs
        :type file_ids: Iterable[str]
        :param target_dir: Directory containing SAFE products
        :type target_dir: str or pathlib.Path
        :return: List of resolved SAFE directory paths
        :rtype: list[pathlib.Path]
        """
        target_dir = Path(target_dir)
        safe_dirs = list(target_dir.glob('*.SAFE'))

        resolved = []
        for fid in file_ids:
            m = re.search(r'_(\d{8}T\d{6})_.*_([A-F0-9]{4})-BURST$', fid)
            if m:
                timestamp, burst_id = m.groups()
                print(timestamp, burst_id)
                match = next(
                    (
                        p
                        for p in safe_dirs
                        if timestamp in p.name and f'_{burst_id}.SAFE' in p.name
                    ),
                    None,
                )
            else:
                match = None
            if match:
                resolved.append(match)
            else:
                raise GaiaDataError(f'SAFE not found for {fid}')

        return resolved
