from earth_observation_data_uploader.data_acquisition_gateway.base_backend import DataAcquisitionBackend

class EODAGDataAcquisitionBackend(DataAcquisitionBackend):
    def __init__(self):
        self._dag = EODataAccessGateway()

    def search(self, provider, start, end, geom, **kwargs) -> SearchResult:
        # self._dag.set_preferred_provider(provider)
        # return dag.search(...)
        raise NotImplementedYet()

    def download(self, product: EOProduct) -> XarrayDict:
        raise NotImplementedYet()
