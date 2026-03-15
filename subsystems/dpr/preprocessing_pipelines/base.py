from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Mapping, Any

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    metadata = {'title': 'unknown', 'abstract': 'unknown'}

    def configure(self, config: Mapping[str, Any]) -> None:
        """Configure pipeline using structured config.

        param Mapping[str, Any]: pipeline configuration in YAML structure
        """
        self._config = config

    @abstractmethod
    def run(self, aoi, start, end, direction, username, password, data_dir, bbox, lidar_file, output_dem, output_landmask) -> None:
        """Execute pipeline.

        param aoi: AOI geometry
        param start: start time for burst search
        param end: end time for burst search
        param direction: orbit direction
        param username: ASF username
        param password: ASF password
        param data_dir: directory with Sentinel-1 data
        param bbox: bounding box of AOI
        param lidar_file: input file with lidar data
        param output_dem: path for resulting output DEM
        param output_landmask: path for output landmask
        """
        pass
