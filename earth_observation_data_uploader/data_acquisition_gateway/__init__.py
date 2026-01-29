from eodag.api.search_result import SearchResult
from eodag_cube.api.product._product import EOProduct
from eodag_cube.types import XarrayDict

class DataAcquisitionGateway:
    """Data Acquisition Gateway module serves as the automated
    ingestion engine for the sub-system.
    """

    def __init__(self, backend: str = "eodag"):
        if backend == "eodag":
            from earth_observation_data_uploader.data_acquisition_gateway.eodag_backend import EODAGDataAcquisitionBackend as DataAcquisitionBackend
        else:
            # raise GAIAConfigurationError(f"Unsupported data acquisition backend: {backend}")
            pass
        self._backend = DataAcquisitionBackend()

    def search(self, provider: str, start: str, end: str, geom: shapely.geometry.base.BaseGeometry, **kwargs) -> SearchResult:
        """Search for data products that match the specified criteria
        across supported providers using selected data acquisition
        backend.

        :param str provider: the provider to be used
        :param str start: TBD
        :param str end: TBD
        :param shapely.geometry.base.BaseGeometry geom: TBD

        For other arguments check the backend:
         - eodag: https://eodag.readthedocs.io/en/stable/api_reference/core.html#eodag.api.core.EODataAccessGateway.search

        :return: a collection of EO products matching the criteria
        :rtype: SearchResult
        """
        return self._backend.search(kwagrs)

    def download(self, product: EOProduct) -> XarrayDict:
        """Download selected data product using selected data
        acquisition backend.

        :param EOProduct product: EO product to be downloaded
        :return: a dictionary of xarray.Dataset
        :rtype: XarrayDict
        """
        return self._backend.download(product)
