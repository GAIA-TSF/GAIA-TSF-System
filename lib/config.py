from typing import Any, Dict, List

import yaml
from pathlib import Path

from shapely import wkt
from shapely.errors import ShapelyError
from osgeo import gdal, osr

gdal.UseExceptions()


class ConfigReader(dict):
    def __init__(self, config_path: str):
        """Initialize config reader.

        :param str config_path: path to config file
        """
        self.config_path = Path(config_path)
        super().__init__()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
            self.update(_config)


class YamlValidator:
    def __init__(self, required_options):
        """Initialize YamlValidator.

        :param dict required_options: required options to be validated
        """
        self.required_options = required_options

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validates YAML dictionary against required_options schema.

        :param Dict[str, Any] config: YAML directory

        :returns list: list of error messages. Empty list = valid.
        """
        self.errors = []
        self._validate_recursive(
            schema=self.required_options, data=config, path='', errors=self.errors
        )
        return self.errors

    def _validate_recursive(
        self, schema: Dict[str, Any], data: Dict[str, Any], path: str, errors: List[str]
    ):
        """
        Recursively validate a configuration dictionary against a required schema.

        The method traverses the provided ``schema`` and verifies that required
        keys exist in ``data``. Nested dictionaries are validated recursively.
        Validation errors are appended to the ``errors`` list instead of raising
        exceptions.

        Validation rules:

        - Missing required keys are reported.
        - Nested dictionaries are validated recursively.
        - If a schema value is ``None``, only key existence is checked.
        - Other schema values may represent additional type or semantic
          constraints enforced by the caller. Currently only WKT is checked.

        :param Dict[str, Any] schema: Dictionary defining required keys and nested structure.

        :param Dict[str, Any] data: Configuration dictionary being validated.

        :param str path: Dot-delimited path of the current validation context,
                 used for error reporting.

        :param List[str] errors: Mutable list used to collect validation error messages.

        :returns: None. Validation errors are accumulated in ``errors``.
        :rtype: None
        """
        for key, opt in schema.items():
            current_path = f'{path}.{key}' if path else key

            # check key
            if key not in data:
                errors.append(f"Missing required key: '{current_path}'")
                continue

            value = data[key]

            # if nested structure is required
            if isinstance(opt, dict):
                if not isinstance(value, dict):
                    errors.append(f"Key '{current_path}' must be a dictionary.")
                else:
                    self._validate_recursive(opt, value, current_path, errors)

            # opt == None -> only check existence
            if value is None:
                errors.append(f"Key '{current_path}' must not be None.")
            if key == 'geom':
                if self._is_valid_wkt(value) is False:
                    errors.append(f"Geometry in '{current_path}' is not valid.")

    @staticmethod
    def _is_valid_wkt(wkt_string: str) -> bool:
        """Check WKT by shapely if it's valid.

        :param str wkt_string: WKT to be validated

        :return: validity flag (True/False)
        :rtype: bool
        """
        try:
            geom = wkt.loads(wkt_string)
            return geom.is_valid
        except (ShapelyError, TypeError):
            return False

    def is_valid(self):
        """Check if config is valid.

        Rules to be checked:
         - No required options are missing
         - Geom WKT valid

        :return: True when config is valid otherwise False
        :rtype: bool
        """
        return len(self.errors) < 1


class ProjectConfigReader(ConfigReader, YamlValidator):
    def __init__(self, config_path: str | Path):
        """Initialize project config reader.

        Validity may be checked by is_valid() method.

        :param str config_path: path to config file
        """
        ConfigReader.__init__(self, config_path)
        YamlValidator.__init__(self, {'project': {'name': None, 'aoi': {'geom': None}}})

        self.validate(dict(self))

    def aoi(self, target_epsg: int = 4326):
        """Get area of interest as WKT string in specified CRS.

        :param int epsg: EPSG code for target CRS

        :return WKT string
        :rtype str
        """
        file_path = self.config_path.parent / self['project']['aoi']
        ds = gdal.OpenEx(file_path, gdal.OF_VECTOR)
        layer = ds.GetLayer(0)

        if layer.GetFeatureCount() > 1:
            raise RuntimeError('AOI: Only one feature expected')

        srs = layer.GetSpatialRef()
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(target_epsg)
        target_srs.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER
        )  # EODAG expects long, lat
        transform = None
        if not srs.IsSame(target_srs):
            transform = osr.CoordinateTransformation(srs, target_srs)

        feature = layer.GetNextFeature()
        if feature is None:
            ds = None
            raise RuntimeError('No features found')

        geom = feature.GetGeometryRef().Clone()
        if transform is not None:
            geom.Transform(transform)

        wkt = geom.ExportToWkt()
        ds = None

        return wkt


class SettingsReader(ConfigReader):
    """Get internal system settings."""

    def __init__(self):
        super().__init__(Path(__file__).parent.parent / 'config.yaml')
