from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eodag.api.search_result import SearchResult
    from eodag_cube.api.product._product import EOProduct
    from eodag_cube.types import XarrayDict

class DataAcquisitionBackend(ABC):
    @abstractmethod
    def search(self, provider, start, end, geom, **kwargs) -> SearchResult:
        pass

    @abstractmethod
    def download(self, product: EOProduct) -> XarrayDict:
        pass
