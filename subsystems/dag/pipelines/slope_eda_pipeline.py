from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
import rasterio
import yaml

from subsystems.dag.core.interfaces import Pipeline
from subsystems.dag.core.registry import PLUGIN_REGISTRY
import subsystems.dag.plugins  # noqa: F401
from subsystems.dag.plugins.eda.slope_eda import SlopeEDA, SlopeEDAResult
from subsystems.dag.plugins.ingestion.sentinel1_loader import Sentinel1LOSLoader
from subsystems.dag.utils.raster import apply_mask, write_single_band_raster


LOGGER = logging.getLogger(__name__)


class SlopeEDAPipeline(Pipeline):
    """Scenario 1 pipeline for Sentinel-1 LOS EDA."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.project_dir = self._project_dir()
        self.loader = self._create_loader()
        self.eda = self._create_eda()

    def run(self) -> dict[str, Any]:
        """Run the Sentinel-1 LOS EDA pipeline."""
        scenario_config = self._scenario_config()
        input_config = scenario_config['inputs']
        static_config = scenario_config['static']
        result_config = scenario_config['results']['eda']

        los_dir = self._resolve_path(input_config['los']['directory'])
        mask_path = self._resolve_path(static_config['tsf_mask'])
        output_dir = self._resolve_path(result_config['output_dir'])
        filename_pattern = str(input_config['los']['filename_pattern'])

        LOGGER.info('Starting slope EDA pipeline.')
        series = self.loader.load(los_dir, filename_pattern)
        mask = self._load_mask(mask_path, series.data.shape[1:])
        masked_data = apply_mask(series.data, mask)

        result = self.eda.run(
            data=masked_data,
            dates=series.dates,
            output_dir=output_dir,
            options=result_config,
        )
        self._write_map_outputs(result, result_config, output_dir, series.profile)

        return {
            'pipeline': 'slope_eda',
            'acquisitions': len(series.dates),
            'output_dir': str(output_dir),
            'statistics': str(result.paths.statistics),
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

    def _create_loader(self) -> Sentinel1LOSLoader:
        plugin = PLUGIN_REGISTRY.create('sentinel1_los_loader')
        if not isinstance(plugin, Sentinel1LOSLoader):
            raise TypeError('sentinel1_los_loader plugin has invalid type.')
        return plugin

    def _create_eda(self) -> SlopeEDA:
        plugin = PLUGIN_REGISTRY.create('slope_eda')
        if not isinstance(plugin, SlopeEDA):
            raise TypeError('slope_eda plugin has invalid type.')
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

    def _write_map_outputs(
        self,
        result: SlopeEDAResult,
        result_config: dict[str, Any],
        output_dir: Path,
        profile: object,
    ) -> None:
        filenames = result_config['filenames']
        if not isinstance(filenames, dict):
            raise ValueError('EDA filenames configuration must be a mapping.')
        raster_format = str(result_config.get('raster_format', 'GTiff'))
        write_single_band_raster(
            output_dir / str(filenames['mean_map']),
            result.mean_map,
            profile,
            driver=raster_format,
        )
        write_single_band_raster(
            output_dir / str(filenames['std_map']),
            result.std_map,
            profile,
            driver=raster_format,
        )
