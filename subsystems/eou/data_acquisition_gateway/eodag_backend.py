from __future__ import annotations
from typing import TYPE_CHECKING

import os
import glob
import rioxarray
from eodag import EODataAccessGateway
if TYPE_CHECKING:
    from eodag.api.search_result import SearchResult
    from eodag_cube.api.product._product import EOProduct
    from eodag_cube.types import XarrayDict

from eou.data_acquisition_gateway.base_backend import DataAcquisitionBackend

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

    def download(self, product: EOProduct, quicklook: bool = False, **kwargs) -> str | XarrayDict:
        if quicklook:
            return product.get_quicklook(**kwargs)

        local_path = self._dag.download(product, extract=True, **kwargs)

        product_data = XarrayDict()
        image_files = glob.glob(os.path.join(local_path, "**", "*.[jt][ip][f]*"), recursive=True)

        for file_path in image_files:
            band_key = os.path.splitext(os.path.basename(file_path))[0]
            try:
                ds = rioxarray.open_rasterio(
                    file_path,
                    chunks={'x': 1024, 'y': 1024}
                ).to_dataset(name="data")
                product_data[band_key] = ds
            except Exception:
                continue

        return product_data
