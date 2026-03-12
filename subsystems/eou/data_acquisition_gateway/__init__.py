from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eodag.api.search_result import SearchResult
    from eodag_cube.api.product._product import EOProduct

from lib.base import GaiaBase, SubsystemId


class DataAcquisitionGateway(GaiaBase):
    """Data Acquisition Gateway module serves as the automated
    ingestion engine for the sub-system.
    """

    def __init__(self, backend: str = 'eodag'):
        super().__init__(SubsystemId.EOU)

        if backend == 'eodag':
            from subsystems.eou.data_acquisition_gateway.eodag_backend import (
                EODAGDataAcquisitionBackend as DataAcquisitionBackend,
            )
        else:
            raise RuntimeError(f'Unsupported data acquisition backend: {backend}')

        self._backend = DataAcquisitionBackend()
        # TBD: raise GaiaSettingsError
        self._backend.set_config(self.settings['eou']['eodag'])

    def search(
        self, provider: str, start: str, end: str, geom: str, **kwargs
    ) -> SearchResult:
        """Search for data products that match the specified criteria
        across supported providers using selected data acquisition
        backend.

        :param str provider: the provider to be used
        :param str start: start date to be used for temporal filter
        :param str end: end date to be used for temporal filer
        :param str geom: geometry as WKT

        For other arguments check the backend:
         - eodag: https://eodag.readthedocs.io/en/stable/api_reference/core.html#eodag.api.core.EODataAccessGateway.search

        :return: a collection of EO products matching the criteria
        :rtype: SearchResult
        """
        self.logger.info(
            f'Search filter: {provider} | {start} | {end} | {geom} | {kwargs}'
        )
        return self._backend.search(provider, start, end, geom, **kwargs)

    def download(self, product: EOProduct, quicklook: bool = False, **kwargs) -> str:
        """Download selected data product using selected data
        acquisition backend.

        :param EOProduct product: EO product to be downloaded
        :param bool quicklook: If True, only download the preview image
        :return: a path to the download data
        :rtype: str
        """
        return self._backend.download(product, quicklook=quicklook, **kwargs)
