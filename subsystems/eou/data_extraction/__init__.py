from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metadata import StacItemFactory

from .metadata import RasterDataset, StacItemFactory

class DataExtraction:
    """Data Extraction module acts as the central logic module for
    ingestion. It receives inputs from both the manual loader and the
    acquisition gateway, performing the necessary extraction and
    preparation steps before handing the data off to the downstream
    Data Processing sub-system.
    """

    def __init__(self, raster_filename: str):
        """
        Initialize DataExctraction module.

        :param str path: Path to the input raster file (GDAL-supported) to be processed
        """
        self._raster_filename = raster_filename

    def stac_factory(self) -> StacItemFactory:
        """
        Creates a STAC factory to generate metadata from input raster datasource.

        :return: STAC factory
        :rtype: StacItemFactory
        """
        return StacItemFactory(RasterDataset(self._raster_filename))
