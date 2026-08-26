from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

from subsystems.dag.core.interfaces import Pipeline
from subsystems.dag.core.registry import PLUGIN_REGISTRY
import subsystems.dag.plugins  # noqa: F401
from subsystems.dag.plugins.features.topographic_features import (
    TopographicFeatureExtractor,
)
from subsystems.dag.utils.io import write_feature_rasters, write_json
from subsystems.dag.utils.raster import RasterProfile
from subsystems.dag.utils.statistics import feature_statistics


LOGGER = logging.getLogger(__name__)


class TopographicFeaturePipeline(Pipeline):
    """Create aligned static DEM, slope, and topographic-position features."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        with config_path.open(encoding='utf-8') as stream:
            self.config = yaml.safe_load(stream)
        if not isinstance(self.config, dict):
            raise ValueError('DAG config must be a mapping.')
        project_dir = self.config.get('project_dir')
        if not isinstance(project_dir, str) or not project_dir.strip():
            raise KeyError('Missing required project_dir in config.yaml.')
        self.project_dir = Path(project_dir).expanduser().resolve()
        extractor = PLUGIN_REGISTRY.create('topographic_feature_extractor')
        if not isinstance(extractor, TopographicFeatureExtractor):
            raise TypeError('topographic_feature_extractor plugin has invalid type.')
        self.extractor = extractor

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_dir / path).resolve()

    def run(self) -> dict[str, Any]:
        settings = self.config.get('static_topography')
        if not isinstance(settings, dict):
            raise ValueError('static_topography must be a mapping.')
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
        output = settings.get('results', {})
        if not isinstance(output, dict):
            raise ValueError('static_topography.results must be a mapping.')
        output_dir = self._resolve(str(output.get('output_dir', 'results/static_features')))
        filenames = output.get('filenames', {})
        if not isinstance(filenames, dict):
            raise ValueError('static_topography.results.filenames must be a mapping.')
        output_paths = write_feature_rasters(
            features,
            output_dir,
            {str(key): str(value) for key, value in filenames.items()},
            profile,
            str(output.get('raster_format', 'GTiff')),
        )
        metadata_path = output_dir / str(output.get('metadata_filename', 'metadata.json'))
        write_json(
            metadata_path,
            {
                'feature_names': sorted(features),
                'creation_date': datetime.now(timezone.utc).isoformat(),
                'source_dem': str(dem_path),
                'pi_definition': 'elevation minus local mean elevation',
                'pi_window_size': pi_window_size,
                'units': {'dem': 'm', 'slope': 'degree', 'pi': 'm'},
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
