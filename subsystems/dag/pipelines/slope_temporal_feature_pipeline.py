from __future__ import annotations

from datetime import date, datetime
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
from subsystems.dag.plugins.features.temporal_features import TemporalFeatureExtractor
from subsystems.dag.plugins.ingestion.sentinel1_loader import Sentinel1LOSLoader
from subsystems.dag.utils.io import write_feature_rasters, write_json
from subsystems.dag.utils.normalization import normalize_features
from subsystems.dag.utils.raster import RasterProfile, apply_mask
from subsystems.dag.utils.statistics import feature_statistics
from subsystems.dag.utils.temporal import temporal_gradient


LOGGER = logging.getLogger(__name__)


class SlopeTemporalFeaturePipeline(Pipeline):
    """Scenario 2b pipeline for generic temporal features."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.project_dir = self._project_dir()
        self.loader = self._create_loader()
        self.extractor = self._create_extractor()

    def run(self) -> dict[str, Any]:
        """Run temporal feature engineering for the slope key variable."""
        LOGGER.info('Processing temporal features.')
        scenario_config = self._scenario_config()
        result_config = scenario_config['results']['temporal_features']
        if not bool(result_config.get('enabled', False)):
            LOGGER.info('Temporal features are disabled in config.')
            return {
                'pipeline': 'slope_temporal_features',
                'features': [],
                'output_dir': None,
                'metadata': None,
            }

        input_config = scenario_config['inputs']
        static_config = scenario_config['static']
        output_dir = self._resolve_path(result_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info('Starting slope temporal feature pipeline.')
        series = self.loader.load(
            self._resolve_path(input_config['los']['directory']),
            str(input_config['los']['filename_pattern']),
        )
        mask = self._load_mask(
            self._resolve_path(static_config['tsf_mask']),
            series.data.shape[1:],
        )
        masked_data = apply_mask(series.data, mask)
        base_stacks = self._build_base_feature_stacks(
            masked_data,
            series.dates,
            result_config,
        )
        temporal_features = self.extractor.compute(
            base_stacks,
            series.dates,
            result_config,
        )
        temporal_features.update(
            self._build_calendar_features(
                dates=series.dates,
                spatial_shape=masked_data.shape[1:],
                mask=mask,
                result_config=result_config,
            ),
        )
        temporal_features = normalize_features(
            temporal_features,
            self.config.get('preprocessing', {}).get('normalization'),
        )
        output_paths = write_feature_rasters(
            features=temporal_features,
            output_dir=output_dir,
            filenames=self._filenames(result_config, temporal_features),
            profile=series.profile,
            raster_format=str(result_config.get('raster_format', 'GTiff')),
            band_names=tuple(
                acquisition_date.isoformat() for acquisition_date in series.dates
            ),
        )
        metadata_path = output_dir / str(result_config['metadata_filename'])
        write_json(
            metadata_path,
            self._build_metadata(
                result_config=result_config,
                base_feature_names=sorted(base_stacks),
                temporal_features=temporal_features,
                output_paths=output_paths,
                input_files=series.source_paths,
                dates=series.dates,
                profile=series.profile,
            ),
        )

        return {
            'pipeline': 'slope_temporal_features',
            'features': sorted(temporal_features),
            'output_dir': str(output_dir),
            'metadata': str(metadata_path),
        }

    def _build_base_feature_stacks(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
        result_config: dict[str, object],
    ) -> dict[str, np.ndarray]:
        configured_features = result_config.get('input_features', ['velocity'])
        if not isinstance(configured_features, list) or not configured_features:
            raise ValueError('temporal_features.input_features must be a list.')

        stacks: dict[str, np.ndarray] = {}
        for feature_name in [str(value) for value in configured_features]:
            if feature_name == 'displacement':
                stacks[feature_name] = data
            elif feature_name == 'velocity':
                stacks[feature_name] = temporal_gradient(data, dates, order=1)
            elif feature_name == 'acceleration':
                stacks[feature_name] = temporal_gradient(data, dates, order=2)
            elif feature_name == 'jerk':
                stacks[feature_name] = temporal_gradient(data, dates, order=3)
            else:
                raise ValueError(f'Unsupported temporal input feature: {feature_name}')
        return stacks

    def _build_calendar_features(
        self,
        dates: tuple[date, ...],
        spatial_shape: tuple[int, int],
        mask: np.ndarray,
        result_config: dict[str, object],
    ) -> dict[str, np.ndarray]:
        """Create date-only annual seasonal feature stacks.

        The values depend solely on the acquisition date, so they are available
        at inference time and do not leak observations from the future.  Each
        one-dimensional calendar signal is broadcast over the TSF raster grid.

        Args:
            dates: Acquisition dates ordered along the temporal axis.
            spatial_shape: Raster height and width.
            mask: TSF mask aligned with the raster grid.
            result_config: Temporal-feature configuration section.

        Returns:
            Mapping of configured calendar feature names to stacks with shape
            ``(time, height, width)``.

        Raises:
            ValueError: If the calendar configuration is invalid.
        """
        calendar_config = result_config.get('calendar', {})
        if not isinstance(calendar_config, dict):
            raise ValueError('temporal_features.calendar must be a mapping.')
        if not bool(calendar_config.get('enabled', False)):
            return {}

        configured_features = calendar_config.get('features', [])
        if not isinstance(configured_features, list) or not configured_features:
            raise ValueError(
                'temporal_features.calendar.features must be a non-empty list '
                'when calendar features are enabled.',
            )

        period_days = calendar_config.get('annual_period_days')
        if not isinstance(period_days, (int, float)) or period_days <= 0:
            raise ValueError(
                'temporal_features.calendar.annual_period_days must be positive.',
            )

        requested_features = [str(feature) for feature in configured_features]
        supported_features = {'annual_sin', 'annual_cos'}
        unsupported = set(requested_features) - supported_features
        if unsupported:
            raise ValueError(
                'Unsupported calendar feature(s): '
                f'{", ".join(sorted(unsupported))}.',
            )

        day_of_year = np.asarray(
            [acquisition_date.timetuple().tm_yday for acquisition_date in dates],
            dtype=np.float64,
        )
        phase = 2.0 * np.pi * (day_of_year - 1.0) / float(period_days)
        feature_values = {
            'annual_sin': np.sin(phase),
            'annual_cos': np.cos(phase),
        }
        tsf_mask = mask.astype(bool, copy=False)
        if tsf_mask.shape != spatial_shape:
            raise ValueError('TSF mask shape does not match the feature grid.')

        features: dict[str, np.ndarray] = {}
        for feature_name in requested_features:
            values = np.broadcast_to(
                feature_values[feature_name][:, np.newaxis, np.newaxis],
                (len(dates), *spatial_shape),
            )
            features[feature_name] = np.where(
                tsf_mask[np.newaxis, :, :],
                values,
                np.nan,
            ).astype(np.float32)
        return features

    def _filenames(
        self,
        result_config: dict[str, object],
        temporal_features: dict[str, np.ndarray],
    ) -> dict[str, str]:
        configured = result_config.get('filenames', {})
        if configured and not isinstance(configured, dict):
            raise ValueError('Temporal feature filenames must be a mapping.')
        filenames = {name: f'{name}.tif' for name in temporal_features}
        filenames.update({str(key): str(value) for key, value in configured.items()})
        return filenames

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

    def _create_loader(self) -> Sentinel1LOSLoader:
        plugin = PLUGIN_REGISTRY.create('sentinel1_los_loader')
        if not isinstance(plugin, Sentinel1LOSLoader):
            raise TypeError('sentinel1_los_loader plugin has invalid type.')
        return plugin

    def _create_extractor(self) -> TemporalFeatureExtractor:
        plugin = PLUGIN_REGISTRY.create('temporal_feature_extractor')
        if not isinstance(plugin, TemporalFeatureExtractor):
            raise TypeError('temporal_feature_extractor plugin has invalid type.')
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
        result_config: dict[str, object],
        base_feature_names: list[str],
        temporal_features: dict[str, np.ndarray],
        output_paths: dict[str, str],
        input_files: tuple[Path, ...],
        dates: tuple[date, ...],
        profile: RasterProfile,
    ) -> dict[str, object]:
        return {
            'feature_names': sorted(temporal_features),
            'base_feature_names': base_feature_names,
            'acquisition_dates': [
                acquisition_date.isoformat() for acquisition_date in dates
            ],
            'creation_date': datetime.now(tz=UTC).isoformat(),
            'processing_parameters': result_config,
            'normalization': self.config.get('preprocessing', {}).get(
                'normalization', {'enabled': False}
            ),
            'input_files': [str(path) for path in input_files],
            'output_files': output_paths,
            'spatial_reference': str(profile.crs),
            'statistics': {
                feature_name: feature_statistics(values)
                for feature_name, values in temporal_features.items()
            },
        }
