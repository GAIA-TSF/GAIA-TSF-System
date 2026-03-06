from __future__ import annotations
from typing import TYPE_CHECKING

import yaml

from eodag import EODataAccessGateway

if TYPE_CHECKING:
    from eodag.api.search_result import SearchResult
    from eodag_cube.api.product._product import EOProduct

from eou.data_acquisition_gateway.base_backend import DataAcquisitionBackend


class EODAGDataAcquisitionBackend(DataAcquisitionBackend):
    def __init__(self):
        self._dag = EODataAccessGateway()

    def search(self, provider, start, end, geom, **kwargs) -> SearchResult:
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
        if quicklook:
            return product.get_quicklook(**kwargs)

        return self._dag.download(product, extract=False, **kwargs)

    def set_config(self, config: dict) -> None:
        self._dag.update_providers_config(yaml.dump(config))
