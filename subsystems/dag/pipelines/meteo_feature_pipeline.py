from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
import csv

import numpy as np
import rasterio
import yaml

from subsystems.dag.core.interfaces import Pipeline
from subsystems.dag.core.registry import PLUGIN_REGISTRY
import subsystems.dag.plugins  # noqa: F401
from subsystems.dag.plugins.features.meteo_features import MeteoFeatureExtractor
from subsystems.dag.plugins.ingestion.meteo_loader import MeteoRasterLoader
from subsystems.dag.utils.io import write_feature_rasters, write_json
from subsystems.dag.utils.missing_values import handle_missing_values
from subsystems.dag.utils.normalization import normalize_features
from subsystems.dag.utils.outliers import transform_outliers
from subsystems.dag.utils.raster import RasterProfile, RasterTimeSeries, apply_mask
from subsystems.dag.utils.statistics import feature_statistics


class MeteoFeaturePipeline(Pipeline):
    """Create meteorological feature rasters from dated input GeoTIFFs."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.project_dir = self._project_dir()
        self.loader = self._create_loader()
        self.extractor = self._create_extractor()

    def run(self) -> dict[str, Any]:
        scenario = self._scenario_config()
        feature_config = self._mapping(scenario, 'feature_engineering')
        inputs = self._mapping(scenario, 'inputs')
        results = self._mapping(scenario, 'results')
        requested_inputs = self._required_inputs(feature_config)

        table_config = inputs.get('table')
        if table_config is not None:
            if not isinstance(table_config, dict):
                raise ValueError('meteorology.inputs.table must be a mapping.')
            series_by_name = self._load_table_inputs(
                table_config, scenario, requested_inputs
            )
        else:
            series_by_name = {
                name: self._load_input(name, inputs) for name in requested_inputs
            }
        reference = next(iter(series_by_name.values()))
        self._validate_series(series_by_name, reference)
        insar_reference = self._load_insar_reference(inputs)
        self.loader.validate_profile(
            reference.profile,
            insar_reference.profile,
            insar_reference.source_paths[0],
        )

        data = {name: series.data for name, series in series_by_name.items()}
        static = scenario.get('static', {})
        mask = None
        if static:
            if not isinstance(static, dict):
                raise ValueError('meteorology.static must be a mapping.')
            mask_value = static.get('tsf_mask')
            if mask_value:
                mask = self._load_mask(
                    self._resolve_path(str(mask_value)), reference.data.shape[1:]
                )
                data = {name: apply_mask(stack, mask) for name, stack in data.items()}

        daily_features = self.extractor.compute(
            data, reference.dates, feature_config
        )
        features = self._sample_on_insar_dates(
            daily_features,
            reference.dates,
            insar_reference.dates,
        )
        features = handle_missing_values(
            features,
            self.config.get('preprocessing', {}).get('missing_values'),
            mask,
        )
        features = transform_outliers(
            features, self.config.get('preprocessing', {}).get('outliers')
        )
        features = normalize_features(
            features, self.config.get('preprocessing', {}).get('normalization')
        )
        output_dir = self._resolve_path(str(results['output_dir']))
        filenames = results.get('filenames', {})
        if not isinstance(filenames, dict):
            raise ValueError('meteorology.results.filenames must be a mapping.')
        output_paths = write_feature_rasters(
            features,
            output_dir,
            {str(key): str(value) for key, value in filenames.items()},
            insar_reference.profile,
            str(results.get('raster_format', 'GTiff')),
            tuple(value.isoformat() for value in insar_reference.dates),
        )
        metadata_path = output_dir / str(
            results.get('metadata_filename', 'metadata.json')
        )
        write_json(
            metadata_path,
            {
                'feature_names': sorted(features),
                'creation_date': datetime.now(tz=timezone.utc).isoformat(),
                'processing_parameters': feature_config,
                'normalization': self.config.get('preprocessing', {}).get(
                    'normalization', {'enabled': False}
                ),
                'missing_values': self.config.get('preprocessing', {}).get(
                    'missing_values', {'enabled': False}
                ),
                'outliers': self.config.get('preprocessing', {}).get(
                    'outliers', {'enabled': False}
                ),
                'dates': [value.isoformat() for value in insar_reference.dates],
                'meteorological_dates': [
                    value.isoformat() for value in reference.dates
                ],
                'temporal_reference': 'insar_acquisitions',
                'input_files': {
                    name: [str(path) for path in series.source_paths]
                    for name, series in series_by_name.items()
                },
                'insar_input_files': [
                    str(path) for path in insar_reference.source_paths
                ],
                'output_files': output_paths,
                'spatial_reference': str(insar_reference.profile.crs),
                'statistics': {
                    name: feature_statistics(values)
                    for name, values in features.items()
                },
            },
        )
        return {
            'pipeline': 'meteo_features',
            'features': sorted(features),
            'output_dir': str(output_dir),
            'metadata': str(metadata_path),
        }

    def _required_inputs(self, config: dict[str, Any]) -> list[str]:
        requested = self.extractor._requested_features(config)
        inputs: list[str] = []
        if requested.intersection(self.extractor.PRECIPITATION_FEATURES):
            inputs.append('precipitation')
        mean_features = {
            'temperature_mean', 'temp_7d_mean', 'temp_30d_mean',
            'temperature_anomaly', 'freezing_degree_days', 'thawing_degree_days',
        }
        if requested.intersection(mean_features):
            inputs.append('temperature_mean')
        if requested.intersection({'temperature_min', 'freeze_thaw'}):
            inputs.append('temperature_min')
        if requested.intersection({'temperature_max', 'freeze_thaw'}):
            inputs.append('temperature_max')
        if not inputs:
            raise ValueError('At least one meteorological feature must be enabled.')
        return inputs

    def _load_insar_reference(
        self, inputs: dict[str, Any]
    ) -> RasterTimeSeries:
        config = self._mapping(inputs, 'insar')
        return self.loader.load(
            self._resolve_path(str(config['directory'])),
            str(config['filename_pattern']),
        )

    def _sample_on_insar_dates(
        self,
        features: dict[str, np.ndarray],
        weather_dates: tuple[date, ...],
        insar_dates: tuple[date, ...],
    ) -> dict[str, np.ndarray]:
        weather_index = {value: index for index, value in enumerate(weather_dates)}
        missing_dates = [value for value in insar_dates if value not in weather_index]
        if missing_dates:
            formatted = ', '.join(value.isoformat() for value in missing_dates[:5])
            suffix = ' ...' if len(missing_dates) > 5 else ''
            raise ValueError(
                'Meteorological data do not cover all InSAR acquisition dates: '
                f'{formatted}{suffix}'
            )
        indices = [weather_index[value] for value in insar_dates]
        return {name: values[indices] for name, values in features.items()}

    def _load_table_inputs(
        self,
        table: dict[str, Any],
        scenario: dict[str, Any],
        requested_inputs: list[str],
    ) -> dict[str, RasterTimeSeries]:
        path = self._resolve_path(str(table['path']))
        if not path.exists():
            raise FileNotFoundError(f'Meteorological table does not exist: {path}')
        columns = table.get('columns', {})
        if not isinstance(columns, dict):
            raise ValueError('meteorology.inputs.table.columns must be a mapping.')
        date_column = str(table.get('date_column', 'date'))

        dates: list[date] = []
        values = {name: [] for name in requested_inputs}
        with path.open('r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or date_column not in reader.fieldnames:
                raise ValueError(f'Missing date column {date_column!r} in {path}.')
            for name in requested_inputs:
                column = str(columns.get(name, name))
                if column not in reader.fieldnames:
                    raise ValueError(f'Missing column {column!r} for {name} in {path}.')
            for row in reader:
                dates.append(self._parse_table_date(row[date_column]))
                for name in requested_inputs:
                    column = str(columns.get(name, name))
                    raw_value = row[column].strip()
                    values[name].append(float(raw_value) if raw_value else np.nan)
        date_tuple = tuple(dates)
        if not date_tuple:
            raise ValueError(f'Meteorological table is empty: {path}')
        date_pairs = zip(date_tuple, date_tuple[1:])
        if any(current <= previous for previous, current in date_pairs):
            raise ValueError(
                'Meteorological table dates must be strictly chronological.'
            )

        static = self._mapping(scenario, 'static')
        mask_path = self._resolve_path(str(static['tsf_mask']))
        if not mask_path.exists():
            raise FileNotFoundError(f'TSF mask does not exist: {mask_path}')
        with rasterio.open(mask_path) as dataset:
            profile = RasterProfile(
                crs=dataset.crs,
                transform=dataset.transform,
                width=dataset.width,
                height=dataset.height,
                dtype='float32',
                nodata=dataset.nodata,
            )
        shape = (len(date_tuple), profile.height, profile.width)
        return {
            name: RasterTimeSeries(
                data=np.broadcast_to(
                    np.asarray(column_values, dtype=np.float32)[:, None, None],
                    shape,
                ).copy(),
                dates=date_tuple,
                profile=profile,
                source_paths=(path,),
            )
            for name, column_values in values.items()
        }

    def _parse_table_date(self, value: str) -> date:
        text = value.strip()
        for date_format in ('%Y%m%d', '%Y-%m-%d'):
            try:
                return datetime.strptime(text, date_format).date()
            except ValueError:
                pass
        raise ValueError(f'Invalid meteorological date: {value!r}')

    def _load_input(
        self, name: str, inputs: dict[str, Any]
    ) -> RasterTimeSeries:
        value = self._mapping(inputs, name)
        return self.loader.load(
            self._resolve_path(str(value['directory'])),
            str(value['filename_pattern']),
        )

    def _validate_series(
        self,
        series_by_name: dict[str, RasterTimeSeries],
        reference: RasterTimeSeries,
    ) -> None:
        for name, series in series_by_name.items():
            if series.dates != reference.dates:
                raise ValueError(f'Meteorological dates do not match for {name}.')
            self.loader.validate_profile(
                reference.profile, series.profile, series.source_paths[0]
            )

    def _load_mask(self, path: Path, shape: tuple[int, int]) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f'TSF mask does not exist: {path}')
        with rasterio.open(path) as dataset:
            mask = dataset.read(1)
        if mask.shape != shape:
            raise ValueError(f'TSF mask dimensions do not match meteo rasters: {path}')
        return mask

    def _load_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f'Config file does not exist: {path}')
        with path.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError('DAG config must be a mapping.')
        return config

    def _project_dir(self) -> Path:
        value = self.config.get('project_dir')
        if not isinstance(value, str) or not value.strip():
            raise KeyError('Missing required project_dir in config.yaml.')
        return Path(value).expanduser().resolve()

    def _scenario_config(self) -> dict[str, Any]:
        return self._mapping(self.config, 'meteorology')

    def _mapping(self, value: dict[str, Any], key: str) -> dict[str, Any]:
        result = value.get(key)
        if not isinstance(result, dict):
            raise ValueError(f'{key} must be a mapping.')
        return result

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_dir / path).resolve()

    def _create_loader(self) -> MeteoRasterLoader:
        plugin = PLUGIN_REGISTRY.create('meteo_raster_loader')
        if not isinstance(plugin, MeteoRasterLoader):
            raise TypeError('meteo_raster_loader plugin has invalid type.')
        return plugin

    def _create_extractor(self) -> MeteoFeatureExtractor:
        plugin = PLUGIN_REGISTRY.create('meteo_feature_extractor')
        if not isinstance(plugin, MeteoFeatureExtractor):
            raise TypeError('meteo_feature_extractor plugin has invalid type.')
        return plugin
