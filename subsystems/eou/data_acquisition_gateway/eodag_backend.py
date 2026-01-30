from earth_observation_data_uploader.data_acquisition_gateway.base_backend import DataAcquisitionBackend
from eodag.api.search_result import SearchResult
from eodag_cube.api.product._product import EOProduct
from eodag_cube.types import XarrayDict
from eodag import EODataAccessGateway

class EODAGDataAcquisitionBackend(DataAcquisitionBackend):
    def __init__(self):
        self._dag = EODataAccessGateway()

    def search(self, provider, start, end, geom, **kwargs) -> SearchResult:
        self._dag.set_preferred_provider(provider)

        # Build search parameters
        search_params = {
            "start": start,
            "end": end,
            "geom": geom,
        }

        search_params.update(kwargs)

        return self._dag.search(**search_params)

    def download(self, product: EOProduct) -> XarrayDict:
        raise NotImplementedYet()
