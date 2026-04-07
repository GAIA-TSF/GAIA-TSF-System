from __future__ import annotations
from typing import TYPE_CHECKING

import yaml

from eodag import EODataAccessGateway

if TYPE_CHECKING:
    from eodag.api.search_result import SearchResult
    from eodag_cube.api.product._product import EOProduct

from subsystems.eou.data_acquisition_gateway.base_backend import DataAcquisitionBackend


class EODAGDataAcquisitionBackend(DataAcquisitionBackend):
    def __init__(self):
        super().__init__()
        self._dag = EODataAccessGateway()

    def set_config(self, config: dict) -> None:
        """Set configuration parameters for eodag backend.

        :param dict config: configuration parameters
        :return: None
        """
        super().set_config(config)
        self._dag.update_providers_config(yaml.dump(config))

    def search(self, provider: str, start: str, end: str, geom: str, **kwargs) -> SearchResult:
        """Search for data products that match the specified criteria
        across supported providers using eodag backend.

        :param str provider: the provider to be used
        :param str start: start date to be used for temporal filter
        :param str end: end date to be used for temporal filer
        :param str geom: geometry as WKT

        For other arguments check the backend:
         - eodag: https://eodag.readthedocs.io/en/stable/api_reference/core.html#eodag.api.core.EODataAccessGateway.search

        :return: a collection of EO products matching the criteria
        :rtype: SearchResult
        """
        self._dag.set_preferred_provider(provider)

        # Build search parameters
        search_params = {
            'start': start,
            'end': end,
            'geom': geom,
        }
        search_params.update(kwargs)

        return self._dag.search(**search_params)

    def download(self, product: EOProduct, quicklook: bool = False, **kwargs) -> str:
        """Download selected data product using eodag backend.

        :param EOProduct product: EO product to be downloaded
        :param bool quicklook: If True, only download the preview image
        :return: a path to the download data
        :rtype: str
        """
        if quicklook:
            return product.get_quicklook(**kwargs)

        return self._dag.download(product, extract=False, **kwargs)
