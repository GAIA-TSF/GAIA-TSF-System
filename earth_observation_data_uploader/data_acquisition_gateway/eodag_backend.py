from earth_observation_data_uploader.data_acquisition_gateway.base_backend import DataAcquisitionBackend

class EODAGDataAcquisitionBackend(DataAcquisitionBackend):
    def search(self) -> SearchResult:
        raise NotImplementedYet()

    def download(self, product: EOProduct) -> XarrayDict:
        raise NotImplementedYet()
