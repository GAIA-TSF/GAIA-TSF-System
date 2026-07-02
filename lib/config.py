from typing import Any, Dict, List

import yaml
import tempfile
from pathlib import Path

from shapely import wkt
from shapely.errors import ShapelyError
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box
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

        self._root = None  # used by __getattr__

    def __getattr__(self, item: str):
        """
        Return a configuration value using attribute-style access.

        The value is retrieved either from the top-level mapping or from the
        configured root section. Dictionary values are automatically wrapped
        into a configuration object.

        :param item: Name of the configuration key.
        :type item: str

        :returns: Configuration value associated with ``item``.
        :rtype: Any

        :raises AttributeError: If the requested key does not exist.
        """
        try:
            value = self[item] if self._root is None else self[self._root][item]
        except KeyError:
            raise AttributeError(item)

        return self._wrap(value)

    def _wrap(self, value: str):
        """
        Recursively wrap dictionary values into ``ConfigNode`` instances.

        Dictionaries are converted to ``ConfigNode`` objects to enable
        attribute-style access. Lists are processed recursively so that any
        nested dictionaries they contain are wrapped as well. All other
        values are returned unchanged.

        :param value: Value to wrap.
        :type value: Any

        :returns: Wrapped value.
        :rtype: Any
        """
        if isinstance(value, dict):
            return ConfigNode(value)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value


class ConfigNode(dict):
    """
    Dictionary wrapper providing attribute-style access to configuration
    values.

    Nested dictionaries are automatically converted to ``ConfigNode``
    instances, allowing recursive access using dot notation. Lists are
    processed so that any dictionary elements they contain are wrapped as
    ``ConfigNode`` objects as well.

    Example::

        cfg = ConfigNode({"database": {"host": "localhost"}})
        cfg.database.host  # returns "localhost"
    """

    def __getattr__(self, item: str):
        """
        Return a configuration value using attribute-style access.

        Nested dictionaries are automatically wrapped into
        ``ConfigNode`` instances. Lists are traversed and any dictionary
        elements are wrapped as well.

        :param item: Name of the configuration key.
        :type item: str

        :returns: Configuration value associated with ``item``.
        :rtype: Any

        :raises AttributeError: If the requested key does not exist.
        """
        try:
            value = self[item]
        except KeyError:
            raise AttributeError(item)

        if isinstance(value, dict):
            return ConfigNode(value)
        if isinstance(value, list):
            return [ConfigNode(v) if isinstance(v, dict) else v for v in value]
        return value


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
            if key == 'aoi':
                if Path(self.config_path.parent / value).exists() is False:
                    errors.append(f"AOI '{current_path}' not found.")
                else:
                    if self._is_valid_geom(self.aoi()) is False:
                        errors.append(f"Geometry in '{current_path}' is not valid.")

    @staticmethod
    def _is_valid_geom(geom: BaseGeometry) -> bool:
        """Check WKT by shapely if it's valid.

        :param BaseGeometry geom: geometry to be validated

        :return: validity flag (True/False)
        :rtype: bool
        """
        try:
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
        YamlValidator.__init__(self, {'project': {'name': None, 'aoi': None}})

        self.validate(dict(self))

        self._root = 'project'  # used by __getattr__

    def aoi(self, target_epsg: int = 4326, roi: bool = False) -> BaseGeometry:
        """
        Get the area of interest (AOI) as a Shapely geometry in the specified CRS.

        The resulting geometry is reprojected to the requested coordinate reference
        system. By default, the full AOI geometry is returned. If ``roi`` is True,
        the bounding box (envelope) of the AOI is returned instead.

        :param target_epsg: EPSG code of the target coordinate reference system
                            (default is 4326, WGS84).
        :type target_epsg: int

        :param roi: If True, return the bounding box (bbox) of the AOI;
                    otherwise return the full AOI geometry.
        :type roi: bool

        :return: Area of interest as a Shapely geometry in the target CRS.
        :rtype: BaseGeometry
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

        aoi_geom = wkt.loads(geom.ExportToWkt())

        ds = None
        if roi:
            return box(*aoi_geom.bounds)

        return aoi_geom


class SettingsReader(ConfigReader):
    """Get internal system settings."""

    def __init__(self):
        super().__init__(Path(__file__).parent.parent / 'config.yaml')

    def temp_file(self, extension: str | None = None) -> Path:
        """
        Create a temporary file in the configured temporary directory.

        The file is created under the ``tmp`` subdirectory of the configured
        data storage location. The returned path includes the requested file
        extension.

        :param extension: File extension including the leading dot
        (e.g. ``'.json'`` or ``'.tif'``) or ``None``.
        :type extension: str

        :returns: Path to the temporary file.
        :rtype: Path
        """
        temp_dir = Path(SettingsReader()['storage']['data_dir']) / 'tmp'
        if not temp_dir.exists():
            temp_dir.mkdir()
        return Path(tempfile.NamedTemporaryFile(dir=temp_dir, suffix=extension).name)
