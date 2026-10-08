"""Generate static DEM, slope, and PI rasters on the source DEM grid."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

import subsystems.dag.plugins  # noqa: F401
from subsystems.dag.core.interfaces import Pipeline
from subsystems.dag.core.registry import PLUGIN_REGISTRY
from subsystems.dag.plugins.features.topographic_features import (
    TopographicFeatureExtractor,
)
from subsystems.dag.utils.io import write_feature_rasters, write_json
from subsystems.dag.utils.missing_values import handle_missing_values
from subsystems.dag.utils.normalization import normalize_features
from subsystems.dag.utils.outliers import transform_outliers
from subsystems.dag.utils.raster import RasterProfile
from subsystems.dag.utils.statistics import feature_statistics

LOGGER = logging.getLogger(__name__)


class TopographicFeaturePipeline(Pipeline):
    """Create DA_R_04 aligned DEM, slope, and topographic-position features."""

    def __init__(self, config_path: Path) -> None:
        """Load DAG configuration and resolve the registered terrain extractor."""
        self.config_path = config_path
        with config_path.open(encoding='utf-8') as stream:
            self.config = yaml.safe_load(stream)
        if not isinstance(self.config, dict):
            raise TypeError('DAG config must be a mapping.')
        project_dir = self.config.get('project_dir')
        if not isinstance(project_dir, str) or not project_dir.strip():
            raise KeyError('Missing required project_dir in config.yaml.')
        self.project_dir = Path(project_dir).expanduser().resolve()
        extractor = PLUGIN_REGISTRY.create('topographic_feature_extractor')
        if not isinstance(extractor, TopographicFeatureExtractor):
            raise TypeError('topographic_feature_extractor plugin has invalid type.')
        self.extractor = extractor

    def _resolve(self, value: str) -> Path:
        """Resolve an absolute path or a path relative to ``project_dir``."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_dir / path).resolve()

    def run(self) -> dict[str, Any]:
        """Read the configured DEM and write preprocessed static features.

        Returns:
            Pipeline name, generated feature names, output directory, and
            metadata path. Output rasters retain the source DEM grid and CRS.

        Raises:
            FileNotFoundError: If the source DEM is absent.
            ValueError: If configuration or raster content is invalid.
        """
        settings = self.config.get('static_topography')
        if not isinstance(settings, dict):
            raise TypeError('static_topography must be a mapping.')
        dem_path = self._resolve(str(settings.get('dem', 'static/tsf_dem.tif')))
        if not dem_path.exists():
            raise FileNotFoundError(f'DEM does not exist: {dem_path}')

        with rasterio.open(dem_path) as dataset:
            dem = dataset.read(1, masked=True).filled(np.nan)
            profile = RasterProfile(
                crs=dataset.crs,
                transform=dataset.transform,
                width=dataset.width,
                height=dataset.height,
                dtype='float32',
                nodata=dataset.nodata,
            )
            pixel_size_x = abs(float(dataset.transform.a))
            pixel_size_y = abs(float(dataset.transform.e))

        pi_window_size = int(settings.get('pi_window_size', 3))
        features = self.extractor.compute(
            dem, pixel_size_x, pixel_size_y, pi_window_size
        )
        features = handle_missing_values(
            features, self.config.get('preprocessing', {}).get('missing_values')
        )
        features = transform_outliers(
            features, self.config.get('preprocessing', {}).get('outliers')
        )
        features = normalize_features(
            features, self.config.get('preprocessing', {}).get('normalization')
        )
        normalization = self.config.get('preprocessing', {}).get('normalization', {})
        normalized = isinstance(normalization, dict) and bool(
            normalization.get('enabled', False)
        )
        outliers = self.config.get('preprocessing', {}).get('outliers', {})
        log_transformed = (
            isinstance(outliers, dict)
            and bool(outliers.get('enabled', False))
            and str(outliers.get('method', 'log')).lower() == 'log'
        )
        output = settings.get('results', {})
        if not isinstance(output, dict):
            raise TypeError('static_topography.results must be a mapping.')
        output_dir = self._resolve(
            str(output.get('output_dir', 'results/static_features'))
        )
        filenames = output.get('filenames', {})
        if not isinstance(filenames, dict):
            raise TypeError('static_topography.results.filenames must be a mapping.')
        output_paths = write_feature_rasters(
            features,
            output_dir,
            {str(key): str(value) for key, value in filenames.items()},
            profile,
            str(output.get('raster_format', 'GTiff')),
        )
        metadata_path = output_dir / str(
            output.get('metadata_filename', 'metadata.json')
        )
        write_json(
            metadata_path,
            {
                'feature_names': sorted(features),
                'creation_date': datetime.now(timezone.utc).isoformat(),
                'source_dem': str(dem_path),
                'pi_definition': 'elevation minus local mean elevation',
                'pi_window_size': pi_window_size,
                'normalization': normalization,
                'missing_values': self.config.get('preprocessing', {}).get(
                    'missing_values', {'enabled': False}
                ),
                'outliers': outliers,
                'units': (
                    {'dem': 'normalized', 'slope': 'normalized', 'pi': 'normalized'}
                    if normalized
                    else (
                        {
                            'dem': 'log-transformed',
                            'slope': 'log-transformed',
                            'pi': 'log-transformed',
                        }
                        if log_transformed
                        else {'dem': 'm', 'slope': 'degree', 'pi': 'm'}
                    )
                ),
                'output_files': output_paths,
                'statistics': {
                    name: feature_statistics(values)
                    for name, values in features.items()
                },
            },
        )
        LOGGER.info('Written static topographic features to %s', output_dir)
        return {
            'pipeline': 'topographic_features',
            'features': sorted(features),
            'output_dir': str(output_dir),
            'metadata': str(metadata_path),
        }
