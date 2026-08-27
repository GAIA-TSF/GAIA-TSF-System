from __future__ import annotations

from datetime import datetime
from datetime import timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any
import logging

import numpy as np
import rasterio
import yaml

from subsystems.dag.core.interfaces import Pipeline
from subsystems.dag.core.registry import PLUGIN_REGISTRY
import subsystems.dag.plugins  # noqa: F401
from subsystems.dag.plugins.features.slope_features import SlopeFeatureExtractor
from subsystems.dag.plugins.ingestion.sentinel1_loader import Sentinel1LOSLoader
from subsystems.dag.utils.io import write_feature_rasters, write_json
from subsystems.dag.utils.normalization import normalize_features
from subsystems.dag.utils.raster import RasterProfile, apply_mask
from subsystems.dag.utils.statistics import feature_statistics


LOGGER = logging.getLogger(__name__)


class SlopeFeaturePipeline(Pipeline):
    """Scenario 2 pipeline for Sentinel-1 LOS slope features."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.project_dir = self._project_dir()
        self.loader = self._create_loader()
        self.extractor = self._create_extractor()

    def run(self) -> dict[str, Any]:
        """Run the slope features pipeline."""
        print('INFO: processing basic slope features.')
        scenario_config = self._scenario_config()
        feature_config = self._feature_config(scenario_config)
        input_config = scenario_config['inputs']
        static_config = scenario_config['static']
        result_config = scenario_config['results']['features']

        los_dir = self._resolve_path(input_config['los']['directory'])
        mask_path = self._resolve_path(static_config['tsf_mask'])
        output_dir = self._resolve_path(result_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info('Starting slope feature pipeline.')
        series = self.loader.load(los_dir, str(input_config['los']['filename_pattern']))
        mask = self._load_mask(mask_path, series.data.shape[1:])
        masked_data = apply_mask(series.data, mask)
        features = self.extractor.compute(masked_data, series.dates, feature_config)
        features = normalize_features(
            features, self.config.get('preprocessing', {}).get('normalization')
        )

        filenames = result_config['filenames']
        if not isinstance(filenames, dict):
            raise ValueError('Feature filenames configuration must be a mapping.')
        output_paths = write_feature_rasters(
            features=features,
            output_dir=output_dir,
            filenames={key: str(value) for key, value in filenames.items()},
            profile=series.profile,
            raster_format=str(result_config.get('raster_format', 'GTiff')),
        )
        metadata_path = output_dir / str(result_config['metadata_filename'])
        write_json(
            metadata_path,
            self._build_metadata(
                feature_config=feature_config,
                features=features,
                output_paths=output_paths,
                input_files=series.source_paths,
                profile=series.profile,
            ),
        )

        return {
            'pipeline': 'slope_features',
            'features': sorted(features),
            'output_dir': str(output_dir),
            'metadata': str(metadata_path),
        }

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f'Config file does not exist: {config_path}')
        with config_path.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError('DAG config must be a mapping.')
        return config

    def _scenario_config(self) -> dict[str, Any]:
        try:
            scenario_config = self.config['slope_stability']
        except KeyError as exc:
            raise KeyError(
                'Missing slope_stability section in config.yaml.',
            ) from exc
        if not isinstance(scenario_config, dict):
            raise ValueError('slope_stability config must be a mapping.')
        return scenario_config

    def _feature_config(self, scenario_config: dict[str, Any]) -> dict[str, bool]:
        try:
            feature_config = scenario_config['feature_engineering']
        except KeyError as exc:
            raise KeyError(
                'Missing slope_stability.feature_engineering section in config.yaml.',
            ) from exc
        if not isinstance(feature_config, dict):
            raise ValueError('slope_stability.feature_engineering must be a mapping.')
        return {
            str(name): bool(enabled)
            for name, enabled in feature_config.items()
            if isinstance(enabled, bool)
        }

    def _create_loader(self) -> Sentinel1LOSLoader:
        plugin = PLUGIN_REGISTRY.create('sentinel1_los_loader')
        if not isinstance(plugin, Sentinel1LOSLoader):
            raise TypeError('sentinel1_los_loader plugin has invalid type.')
        return plugin

    def _create_extractor(self) -> SlopeFeatureExtractor:
        plugin = PLUGIN_REGISTRY.create('slope_feature_extractor')
        if not isinstance(plugin, SlopeFeatureExtractor):
            raise TypeError('slope_feature_extractor plugin has invalid type.')
        return plugin

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (self.project_dir / path).resolve()

    def _project_dir(self) -> Path:
        project_dir = self.config.get('project_dir')
        if not isinstance(project_dir, str) or not project_dir.strip():
            raise KeyError('Missing required project_dir in config.yaml.')
        return Path(project_dir).expanduser().resolve()

    def _load_mask(
        self,
        mask_path: Path,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        if not mask_path.exists():
            raise FileNotFoundError(f'TSF mask does not exist: {mask_path}')
        with rasterio.open(mask_path) as dataset:
            mask = dataset.read(1)

        if mask.shape != expected_shape:
            raise ValueError(
                'TSF mask dimensions do not match LOS raster dimensions: '
                f'{mask.shape} != {expected_shape}',
            )
        if not np.any(mask.astype(bool)):
            raise ValueError('TSF mask does not contain any selected pixels.')
        return mask

    def _build_metadata(
        self,
        feature_config: dict[str, bool],
        features: dict[str, np.ndarray],
        output_paths: dict[str, str],
        input_files: tuple[Path, ...],
        profile: RasterProfile,
    ) -> dict[str, object]:
        return {
            'feature_names': sorted(features),
            'creation_date': datetime.now(tz=UTC).isoformat(),
            'processing_parameters': feature_config,
            'normalization': self.config.get('preprocessing', {}).get(
                'normalization', {'enabled': False}
            ),
            'input_files': [str(path) for path in input_files],
            'output_files': output_paths,
            'spatial_reference': str(profile.crs),
            'statistics': {
                feature_name: feature_statistics(values)
                for feature_name, values in features.items()
            },
        }
