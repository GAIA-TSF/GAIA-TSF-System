from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, List, Dict, Any
    from subsystems.qcl.logger import Logger

import os
import json
from pathlib import Path
from stactools.sentinel2.stac import create_item

from osgeo import gdal

gdal.UseExceptions()

from lib.base import GaiaBase, SubsystemId


class RasterDataset:
    """
    GDAL dataset wrapper for extracting spatial and raster metadata.

    :param str path: Path to the raster data source

    :raises RuntimeError: If the file cannot be opened.
    """

    def __init__(self, path: str):
        self.path = path

    @property
    def stac_item(self) -> Dict[str, Any]:
        """Get the STAC item representation for this path.

        :return: A dictionary representing the STAC item with
        """
        return self.get_stac_item(self.path)

    def get_stac_item(self, path: str) -> Dict[str, Any]:
        """Retrieve and transform a STAC item from the given path.

        This method creates a standard STAC item using the stactools library,
        then applies custom transformations to match the desired format for
        Sentinel-2 L2A data.

        :param path: The file path to the granule or product directory.
        :return: A dictionary containing the transformed STAC item with band
            assets, metadata assets, and updated properties.
        """
        # common STAC definition
        item_standard = create_item(
            granule_href=str(path),
            additional_providers=None,
            tolerance=None,
            asset_href_prefix=None,
        )
        item_standard.set_self_href(os.path.split(path)[0])
        # make HREFs relative; otherwise, it wouldn't be possible to test it
        # locally (comparison would fail for HREFs)
        item_standard.make_asset_hrefs_relative()
        item_dict_standard = item_standard.to_dict()

        # now our tweaks
        item_dict_transformed = self._transform_item(item_dict_standard)

        return item_dict_transformed

    @staticmethod
    def _transform_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a stactools Sentinel-2 item to the desired format.

        This method reshapes a standard Sentinel-2 STAC item by:
        - Remapping band assets from semantic names (e.g., 'nir') to band
          numbers (e.g., 'B08')
        - Collecting EO band information as a section separate from raster:bands
        - Updating properties with processing-level and tile information

        :param item: The dictionary representation of a standard STAC item as
            produced by stactools.
        :return: A transformed STAC item dictionary with

        TODO: Probably going to be renamed once support for S1 will be on the table
        """

        # Mapping from semantic band names to band numbers
        band_mapping = {
            'coastal': 'B01',
            'blue': 'B02',
            'green': 'B03',
            'red': 'B04',
            'rededge1': 'B05',
            'rededge2': 'B06',
            'rededge3': 'B07',
            'nir': 'B08',
            'nir08': 'B8A',
            'nir09': 'B09',
            # band 10 missing
            'swir16': 'B11',
            'swir22': 'B12',
        }

        # create new assets dictionary with band number keys
        new_assets = {}
        eo_bands = []

        for asset_key, asset_value in item.get('assets', {}).items():
            # metadata assets section
            if asset_key in ['product_metadata', 'granule_metadata', 'safe_manifest']:
                role = 'metadata'
                if asset_key == 'product_metadata':
                    # we want it called scene_metadata, not product_metadata
                    new_key = 'scene_metadata'
                else:
                    new_key = asset_key

                new_assets[new_key] = {
                    'href': asset_value.get('href'),
                    'type': asset_value.get('type'),
                    'roles': [role],
                }

            # bands assets section
            if asset_key not in band_mapping.keys():
                continue

            band_info = asset_value['eo:bands'][0]
            band_name = band_info.get('name')

            # Create new asset
            new_asset = {
                'href': asset_value.get('href'),
                'type': asset_value.get('type', 'image/jp2'),
                'roles': asset_value.get('roles', ['data']),
            }

            # Use the band name as key
            new_assets[band_name] = new_asset

            eo_bands.append(
                {
                    'name': band_name,
                    'common_name': band_info.get('common_name'),
                    'center_wavelength': band_info.get('center_wavelength'),
                    'gsd': asset_value.get('gsd'),
                }
            )

        # Build the new item
        new_item = {
            'type': item.get('type'),
            'stac_version': '1.0.0',
            'id': item.get('id'),
            'bbox': item.get('bbox'),
            'geometry': item.get('geometry'),
            'properties': {
                'datetime': item.get('properties', {}).get('datetime'),
                'start_datetime': item.get('properties', {}).get('datetime'),
                'end_datetime': item.get('properties', {}).get('datetime'),
                'platform': item.get('properties', {}).get('platform'),
                'constellation': item.get('properties', {}).get('constellation'),
                'instruments': item.get('properties', {}).get('instruments'),
                'processing:level': 'L2A',
                'eo:cloud_cover': item.get('properties', {}).get('eo:cloud_cover'),
                'view:sun_azimuth': item.get('properties', {}).get('view:sun_azimuth'),
                'view:sun_elevation': item.get('properties', {}).get(
                    'view:sun_elevation'
                ),
                'sat:relative_orbit': item.get('properties', {}).get(
                    'sat:relative_orbit'
                ),
                's2:tile_id': item.get('properties', {}).get('grid:code').split('-')[1],
                's2:product_uri': item.get('properties', {}).get('s2:product_uri'),
            },
            'assets': new_assets,
            'collection': 'sentinel-2-l2a',
            'links': [
                {
                    'rel': 'self',
                    'href': 'https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items/'
                    + item.get('id'),
                }
            ],
            'eo:bands': sorted(eo_bands, key=lambda x: x['name']),
            'providers': [
                {
                    'name': provider.get('name'),
                    'roles': provider.get('roles'),
                }
                for provider in item.get('properties', {}).get('providers', [])
            ],
        }

        return new_item


class InsituDataset:
    """
    Wrapper for in-situ CSV datasets providing spatial and temporal metadata
    for STAC item generation. Relies on the metadata dict provided by ISU,
    since CSV files do not embed spatial information like GeoTIFF.

    :param str path: Path to the CSV file
    :param dict metadata: Metadata dict from ISU containing time_range, location, schema, crs, sensor_type
    """

    def __init__(self, path: str, metadata: Dict[str, Any]):
        self.path = Path(path)
        self._meta = metadata
        self._data = metadata.get('data', {})

    @property
    def epsg(self) -> int:
        crs = self._data.get('crs', 'EPSG:4326')
        return int(crs.split(':')[1])

    def get_bbox(self) -> Optional[List[float]]:
        """Returns [min_lon, min_lat, max_lon, max_lat] or None."""
        loc = self._data.get('location')
        return loc['bbox'] if loc else None

    def get_time_range(self) -> Optional[Dict[str, str]]:
        """Returns {'start': ..., 'end': ...} or None."""
        return self._data.get('time_range')

    def get_schema(self) -> List[str]:
        """Returns list of column names."""
        return self._data.get('schema', [])

    def get_sensor_type(self) -> Optional[str]:
        """Returns sensor type string or None."""
        return self._data.get('sensor_type')


class InsituStacItemFactory:
    """
    Creates STAC Item metadata from an InsituDataset.
    """

    STAC_EXTENSIONS: List[str] = [
        'https://stac-extensions.github.io/table/v1.2.0/schema.json',
        'https://stac-extensions.github.io/projection/v1.0.0/schema.json',
    ]

    def __init__(self, dataset: InsituDataset, logger: Any):
        self.dataset = dataset
        self.logger = logger

    def _build_geometry(self, bbox: List[float]) -> Dict[str, Any]:
        return {
            'type': 'Polygon',
            'coordinates': [
                [
                    [bbox[0], bbox[1]],
                    [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]],
                    [bbox[0], bbox[1]],
                ]
            ],
        }

    def create_item(self) -> Dict[str, Any]:
        """
        Creates a STAC Item as a dictionary.

        :return: STAC Item
        :rtype: dict
        """
        bbox = self.dataset.get_bbox()
        time_range = self.dataset.get_time_range()
        schema = self.dataset.get_schema()

        properties: Dict[str, Any] = {
            'datetime': time_range['start'] if time_range else None,
            'start_datetime': time_range['start'] if time_range else None,
            'end_datetime': time_range['end'] if time_range else None,
            'proj:epsg': self.dataset.epsg,
        }
        sensor_type = self.dataset.get_sensor_type()
        if sensor_type:
            properties['sensor_type'] = sensor_type

        stac_item: Dict[str, Any] = {
            'type': 'Feature',
            'stac_version': '1.0.0',
            'stac_extensions': self.STAC_EXTENSIONS,
            'id': self.dataset.path.stem,
            'collection': 'undefined',
            'properties': properties,
            'geometry': self._build_geometry(bbox) if bbox else None,
            'bbox': bbox,
            'assets': {
                'data': {
                    'href': self.dataset.path.name,
                    'type': 'text/csv',
                    'roles': ['data'],
                    'table:columns': [{'name': c} for c in schema],
                }
            },
            'links': [],
        }
        self.logger.debug(f'STAC item created: {stac_item}')
        return stac_item

    def save(self, output_path: str) -> str:
        """
        Saves the STAC Item to a JSON file.

        :param str output_path: Path to the output JSON
        :return: output path
        :rtype: str
        """
        item = self.create_item()
        with open(output_path, 'w') as f:
            json.dump(item, f, indent=4)
        self.logger.info(f'STAC item saved: {output_path}')
        return output_path


class StacItemFactory:
    """
    Creates STAC Item metadata from a RasterDataset.
    """

    def __init__(self, raster: RasterDataset, logger: Logger):
        """
        Initialize StacItemFactory.

        :param RasterDataset raster: RasterDataset instance
        :param Logger logger: specified logger to be used
        """
        self.raster = raster
        self.logger = logger

    def create_item(self) -> Dict[str, Any]:
        """
        Creates a STAC Item as a dictionary.

        :return: STAC Item
        :rtype: dict
        """
        stac_item = self.raster.stac_item

        self.logger.debug(f'STAC item created: {stac_item}')

        return stac_item

    def save(self, output_path: str) -> str:
        """
        Saves the STAC Item to a JSON file.

        :param str output_path: Path to the output JSON

        :return: output path
        :rtype: str
        """
        item = self.create_item()
        with open(output_path, 'w') as f:
            json.dump(item, f, indent=4)
        self.logger.info(f'STAC item saved: {output_path}')

        return output_path


class MetadataGenerator(GaiaBase):
    """
    The automatic generation of metadata during ingestion.
    """

    def __init__(self):
        """Initialize metadata generator."""
        super().__init__(SubsystemId.DPR)
        self._isu_metadata: Dict[str, Any] = {}

    def set_isu_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store ISU-provided metadata for use when processing CSV datasources.

        :param dict metadata: Metadata dict from ISU (time_range, location, schema, crs, sensor_type).
        """
        self._isu_metadata = metadata

    def set_datasource(self, data_source: str):
        """Set the data source for which metadata should be generated.

        :param str data_source: path to the datasource (raster, tabular data...)
        """
        if Path(data_source).suffix in ('.csv',):
            self.logger.info(f'Metadata generator datasource: {data_source}')
            self._ds = InsituDataset(data_source, self._isu_metadata)
            self._factory = InsituStacItemFactory(self._ds, self.logger)
        else:
            self.logger.info(f'Metadata generator datasource: {data_source}')
            self._ds = RasterDataset(data_source)
            self._factory = StacItemFactory(self._ds, self.logger)

    @property
    def stac(self) -> StacItemFactory:
        """
        Creates a STAC factory to generate metadata from input raster datasource.

        :return: STAC factory
        :rtype: StacItemFactory
        """
        return self._factory
