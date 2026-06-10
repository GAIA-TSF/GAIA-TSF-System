from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, List
    from eodag.api.product import EOProduct

import yaml
from pathlib import Path

import zipfile
from shapely.geometry.base import BaseGeometry
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

    def search(
        self, geom: str | BaseGeometry, start: str, end: str, provider: str, **kwargs
    ) -> SearchResult:
        """Search for data products that match the specified criteria
        across supported providers using eodag backend.

        :param str | BaseGeometry geom: geometry as WKT or shapely BaseGeometry object
        :param str start: start date to be used for temporal filter
        :param str end: end date to be used for temporal filer
        :param str provider: the provider to be used

        For other arguments check the backend:
         - eodag: https://eodag.readthedocs.io/en/stable/api_reference/core.html#eodag.api.core.EODataAccessGateway.search

        :return: a collection of EO products matching the criteria
        :rtype: SearchResult
        """
        self._dag.set_preferred_provider(provider)

        # Build search parameters
        search_params = {
            'geom': geom,
            'start': start,
            'end': end,
        }
        search_params.update(kwargs)

        return self._dag.search(**search_params)

    def _download(
        self, product: EOProduct, target_dir: str, quicklook: bool = False, **kwargs
    ) -> Path:
        """Download selected data product using eodag backend.

        :param EOProduct product: EO product to be downloaded
        :param str target_dir: target directory to store downloaded product
        :param bool quicklook: If True, only download the preview image
        :return: a path to the download data
        :rtype: Path
        """
        if quicklook:
            return Path(product.get_quicklook(output_dir=target_dir, **kwargs))

        existing, missing = self.__split_products([product], target_dir)
        if existing:
            return self.__unzip_safe(existing[0][1])
        else:
            return self.__unzip_safe(
                Path(
                    self._dag.download(
                        product, extract=False, output_dir=target_dir, **kwargs
                    )
                )
            )

    def _download_all(
        self, products: SearchResult, target_dir: str, quicklook: bool = False, **kwargs
    ) -> List[Path]:
        """Download all selected data products using eodag backend.

        :param SearchResult products: EO products to be downloaded
        :param str target_dir: target directory to store downloaded products
        :param bool quicklook: If True, only download the preview images
        :return: a path to the download data
        :rtype: Path
        """
        downloaded_paths = []

        if quicklook:
            for product in products:
                downloaded_paths.append(
                    Path(product.get_quicklook(output_dir=target_dir, **kwargs))
                )
            return downloaded_paths

        existing, missing = self.__split_products(products, target_dir)

        if missing:
            downloaded_paths = self._dag.download_all(
                missing, extract=False, output_dir=target_dir, **kwargs
            )

        all_paths = [p for _, p in existing] + [Path(p) for p in downloaded_paths]

        return [self.__unzip_safe(p) for p in all_paths]

    def __split_products(
        self,
        products: Iterable[EOProduct],
        target_dir: str | Path,
    ) -> tuple[list[tuple[EOProduct, Path]], list[EOProduct]]:
        """
        Split products into existing and missing based on local files.

        Existing files are detected as ``<target_dir>/<title>.zip`` and reused
        by updating ``product.location`` to a ``file://`` URI.

        .. note::
        EODAG cache (``.downloaded``) may be inconsistent across runs
        or containers. This function treats the filesystem as the source
        of truth to avoid redundant downloads.

        :param products: Iterable of EODAG product objects
        :type products: Iterable[eodag.api.product.EOProduct]
        :param target_dir: Directory with product ZIP files
        :type target_dir: str or pathlib.Path
        :return: Tuple of (existing, missing)
        :rtype: tuple[list[tuple[eodag.api.product.EOProduct, pathlib.Path]], list[eodag.api.product.EOProduct]]
        """
        existing = []
        missing = []

        for product in products:
            path = Path(target_dir) / f'{product.properties["title"]}.zip'

            if path.exists() and path.stat().st_size > 0:
                # reuse existing file
                product.location = f'file://{path}'
                existing.append((product, path))
            else:
                missing.append(product)

        return existing, missing

    def __unzip_safe(self, zip_path: Path) -> Path:
        """
        Safely extract a ZIP archive into a ``.SAFE`` directory.

        If the target directory already exists, extraction is skipped and the
        existing path is returned. Otherwise, the archive is extracted into
        its parent directory.

        .. note::
        ``extract=True`` is not used because downstream tools (e.g. stactools)
        require a valid ``.SAFE`` structure. Extraction is handled explicitly

        :param zip_path: Path to the input ZIP file
        :type zip_path: pathlib.Path
        :return: Path to the extracted ``.SAFE`` directory
        :rtype: pathlib.Path
        """
        out_dir = Path(zip_path.parent, zip_path.stem + '.SAFE')
        if out_dir.exists():
            return out_dir

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir.parent)
        return out_dir
