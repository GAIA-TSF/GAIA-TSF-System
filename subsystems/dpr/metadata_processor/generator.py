from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, List, Dict, Any

import os
import json
from datetime import datetime, UTC
from pathlib import Path

from osgeo import gdal, osr

gdal.UseExceptions()


class RasterDataset:
    """
    GDAL dataset wrapper for extracting spatial and raster metadata.

    :param str path: Path to the raster data source

    :raises RuntimeError: If the file cannot be opened.
    """

    def __init__(self, path: str):
        self.path = path
        self.dataset = gdal.Open(path)

    @property
    def driver(self) -> str:
        """Returns the GDAL driver name."""
        return self.dataset.GetDriver().ShortName

    @property
    def size(self) -> (int, int):
        """Returns raster dimensions (width, height)."""
        return self.dataset.RasterXSize, self.dataset.RasterYSize

    @property
    def geotransform(self) -> tuple:
        """Returns GDAL geotransform tuple."""
        return self.dataset.GetGeoTransform()

    @property
    def projection(self) -> str:
        """Returns WKT projection string of the dataset."""
        return self.dataset.GetProjection()

    def get_spatial_reference(self) -> Optional[osr.SpatialReference]:
        """
        Returns SpatialReference object for the dataset if available.

        :return: SpatialReference or None
        :rtype: osr.SpatialReference | None
        """
        if not self.projection:
            return None
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.projection)
        return srs

    def get_epsg(self) -> Optional[int]:
        """
        Returns EPSG code if available.

        :return: EPSG code or None
        :rtype: int | None
        """
        srs = self.get_spatial_reference()
        if srs:
            auth = srs.GetAttrValue('AUTHORITY', 1)
            if auth:
                return int(auth)
        return None

    def get_bbox_native(self) -> List[float]:
        """
        Returns bounding box in native dataset coordinates.

        :return: [minx, miny, maxx, maxy]
        :rtype: list[float]
        """
        gt = self.geotransform
        width, height = self.size

        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * width
        miny = maxy + gt[5] * height

        return [minx, miny, maxx, maxy]

    def get_bbox_wgs84(self) -> List[float]:
        """
        Returns bounding box transformed to EPSG:4326 (WGS84).

        :return: [min_lon, min_lat, max_lon, max_lat]
        :rtype: list[float]
        """
        minx, miny, maxx, maxy = self.get_bbox_native()

        source_srs = self.get_spatial_reference()
        if source_srs is None:
            return [minx, miny, maxx, maxy]

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)

        transform = osr.CoordinateTransformation(source_srs, target_srs)

        corners = [
            transform.TransformPoint(minx, miny),
            transform.TransformPoint(minx, maxy),
            transform.TransformPoint(maxx, maxy),
            transform.TransformPoint(maxx, miny),
        ]

        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]

        return [min(xs), min(ys), max(xs), max(ys)]

    def get_band_metadata(self) -> List[Dict[str, Any]]:
        """
        Returns metadata for all raster bands.

        :return: List of dicts containing 'data_type' and 'nodata'
        :rtype: list[dict[str, Any]]
        """
        bands = []
        for i in range(1, self.dataset.RasterCount + 1):
            band = self.dataset.GetRasterBand(i)
            bands.append(
                {
                    'data_type': gdal.GetDataTypeName(band.DataType),
                    'nodata': band.GetNoDataValue(),
                }
            )
        return bands


class StacItemFactory:
    """
    Creates STAC Item metadata from a RasterDataset.
    """

    MIME_LOOKUP: Dict[str, str] = {
        'GTiff': 'image/tiff; application=geotiff',
        'JP2OpenJPEG': 'image/jp2',
    }

    def __init__(self, raster: RasterDataset):
        """
        :param RasterDataset raster: RasterDataset instance
        """
        self.raster: RasterDataset = raster

    def _build_geometry(self, bbox: List[float]) -> Dict[str, Any]:
        """
        Creates GeoJSON Polygon for the STAC Item.

        :param list[float] bbox: [minx, miny, maxx, maxy]

        :return: GeoJSON Polygon
        :rtype: dict
        """
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
        item_id = self.raster.path.stem
        now = datetime.now(UTC).isoformat() + 'Z'

        bbox = self.raster.get_bbox_wgs84()
        geometry = self._build_geometry(bbox)

        width, height = self.raster.size

        stac_item: Dict[str, Any] = {
            'type': 'Feature',
            'stac_version': '1.0.0',
            'stac_extensions': [
                'https://stac-extensions.github.io/projection/v1.0.0/schema.json',
                'https://stac-extensions.github.io/raster/v1.1.0/schema.json',
            ],
            'id': item_id,
            'properties': {
                'datetime': now,
                'proj:epsg': self.raster.get_epsg(),
                'proj:shape': [height, width],
                'proj:transform': list(self.raster.geotransform),
                'raster:bands': self.raster.get_band_metadata(),
            },
            'bbox': bbox,
            'geometry': geometry,
            'assets': {
                'data': {
                    'href': self.raster.path.name,
                    'type': self.MIME_LOOKUP.get(
                        self.raster.driver, 'application/octet-stream'
                    ),
                    'roles': ['data'],
                }
            },
            'links': [],
        }

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
        return output_path


class MetadataGenerator:
    """
    The automatic generation of metadata during ingestion.
    """

    def __init__(self, data_source: str):
        """Initialize Metadata Generator.

        :param str data_source: path to the datasource (raster, tabular data...)
        """
        # TODO: allowed file extensions should be part of internal settings (see #89)
        # TODO: do we want to rely on file extension only?
        if Path(data_source).suffix in ('.tif', '.jp2'):
            self._ds = RasterDataset(data_source)
            self._factory = StacItemFactory(self._ds)
        elif Path(data_source).suffix in ('.csv',):
            # TODO: ISU
            pass
        else:
            # TODO: replace by GAIA-TSF exception
            raise RuntimeError('Unsupported datasource')

    @property
    def stac(self) -> StacItemFactory:
        """
        Creates a STAC factory to generate metadata from input raster datasource.

        :return: STAC factory
        :rtype: StacItemFactory
        """
        return self._factory
