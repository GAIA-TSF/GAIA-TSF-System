"""Cloud-filtered AMD base, temporal, and meteorological feature stages."""

import csv
from copy import deepcopy
import json
from datetime import date

import numpy as np
import rasterio

from subsystems.dag.pipelines.amd_eda_pipeline import AMDEDAPipeline
from subsystems.dag.plugins.features.amd_temporal_features import compute_amd_temporal
from subsystems.dag.plugins.features.meteo_features import MeteoFeatureExtractor
from subsystems.dag.utils.io import write_feature_rasters, write_json
from subsystems.dag.utils.raster import RasterProfile


class AMDFeaturePipeline(AMDEDAPipeline):
    """Produce a named AMD feature and its common accepted acquisition axis."""

    def _name(self):
        return 'amd_difference' if self.options['feature_engineering']['method'] == 'difference' else 'amd_index'

    def run(self):
        original = self.options
        self.options = deepcopy(original)
        raw_directory = self._path(original['results']['features']['output_dir']) / 'source_index'
        self.options['results']['features']['output_dir'] = str(raw_directory)
        self.options['results']['index'] = {'output_dir': str(raw_directory)}
        try:
            result = self._run_index_generation()
        finally:
            self.options = original
        with rasterio.open(result['index']) as source:
            data = source.read(masked=True).filled(np.nan)
            dates = tuple(date.fromisoformat(d) for d in source.descriptions)
            profile = RasterProfile(source.crs, source.transform, source.width,
                                    source.height, 'float32', np.nan)
        domain = np.zeros(data.shape[1:], dtype=bool)
        for region in ('amd', 'clean_water'):
            with rasterio.open(self._path(self.options['static'][f'{region}_mask'])) as source:
                domain |= source.read(1, masked=True).filled(0) == 1
        minimum = float(self.options.get('acquisitions', {}).get('min_valid_fraction', 0))
        if not np.isfinite(minimum) or not 0 <= minimum <= 1:
            raise ValueError('min_valid_fraction must be between zero and one.')
        coverage = np.isfinite(data[:, domain]).mean(axis=1)
        keep = (coverage > 0) & (coverage >= minimum)
        if not keep.any():
            raise ValueError('No cloud-free AMD observations satisfy coverage selection.')
        source_metadata = json.loads((raw_directory / 'metadata.json').read_text())
        selected_dates = tuple(d for d, accepted in zip(dates, keep) if accepted)
        return self._write('features', {self._name(): data[keep]}, selected_dates, profile, {
            'source_paths': [p for p, accepted in zip(source_metadata['source_paths'], keep) if accepted],
            'excluded_dates': [d.isoformat() for d, accepted in zip(dates, keep) if not accepted],
            'valid_fraction': coverage[keep].tolist(),
        })

    def _write(self, stage, features, dates, profile, extra=None):
        options = self.options['results'][stage]
        directory = self._path(options['output_dir'])
        paths = write_feature_rasters(features, directory, options.get('filenames', {}),
                                      profile, 'GTiff', tuple(d.isoformat() for d in dates))
        metadata = directory / 'metadata.json'
        write_json(metadata, {
            'feature_names': list(features), 'acquisition_dates': [d.isoformat() for d in dates],
            'formula': self.options['feature_engineering'],
            'acquisition_selection': self.options.get('acquisitions', {}),
            'quality_filter': self.options.get('quality', {}),
            'spatial_reference': str(profile.crs), 'output_files': paths,
            **(extra or {}),
        })
        return {'pipeline': f'amd_{stage}', 'features': list(features),
                'output_files': paths, 'metadata': str(metadata), 'acquisitions': len(dates)}

    def _base(self):
        directory = self._path(self.options['results']['features']['output_dir'])
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['formula'] != self.options['feature_engineering'] or metadata.get('acquisition_selection') != self.options.get('acquisitions', {}) or metadata.get('quality_filter') != self.options.get('quality', {}):
            raise ValueError('AMD configuration changed; rerun amd_features first.')
        with rasterio.open(metadata['output_files'][self._name()]) as source:
            data = source.read(masked=True).filled(np.nan)
            dates = tuple(date.fromisoformat(d) for d in source.descriptions)
            profile = RasterProfile(source.crs, source.transform, source.width,
                                    source.height, 'float32', np.nan)
        return data, dates, profile


class AMDTemporalFeaturePipeline(AMDFeaturePipeline):
    def run(self):
        data, dates, profile = self._base()
        config = self.options.get('temporal_features', {})
        features = compute_amd_temporal(data, dates, config)
        return self._write('temporal_features', features, dates, profile, {
            'processing_parameters': config, 'base_feature': self._name(),
            'history': 'previous valid observations per pixel; no yearly reset',
            'standard_deviation_ddof': 0,
        })


class AMDMeteoFeaturePipeline(AMDFeaturePipeline):
    def run(self):
        data, dates, profile = self._base()
        config = self.options['meteorology']
        table = config['inputs']['table']
        engineering = config['feature_engineering']
        if engineering.get('temperature_anomaly') and 'temperature_baseline' not in engineering:
            raise ValueError('AMD temperature_anomaly requires an explicit temperature_baseline.')
        columns = table.get('columns', {})
        weather_dates, rows = [], []
        with self._path(table['path']).open(newline='') as stream:
            for row in csv.DictReader(stream):
                weather_dates.append(date.fromisoformat(row[table.get('date_column', 'date')]))
                rows.append(row)
        if not weather_dates or any((b - a).days != 1 for a, b in zip(weather_dates, weather_dates[1:])):
            raise ValueError('AMD meteorology requires a consecutive chronological daily table.')
        names = set(MeteoFeatureExtractor.PRECIPITATION_FEATURES + MeteoFeatureExtractor.TEMPERATURE_FEATURES + MeteoFeatureExtractor.COLD_REGION_FEATURES)
        unknown = {k for k, v in engineering.items() if v is True} - names - {'cold_regions'}
        if unknown:
            raise ValueError(f'Unknown weather features: {unknown}')
        weather = {}
        for name in ('precipitation', 'temperature_mean', 'temperature_min', 'temperature_max'):
            column = columns.get(name, name)
            if column in rows[0]:
                weather[name] = np.array([float(r[column]) if r[column].strip() else np.nan for r in rows], dtype=np.float32)[:, None, None]
        # Calculate weather on a single daily series, then broadcast only sampled results.
        daily = MeteoFeatureExtractor().compute(weather, tuple(weather_dates), engineering)
        if not daily:
            raise ValueError('Enable at least one AMD meteorological feature.')
        index = {d: i for i, d in enumerate(weather_dates)}
        if any(d not in index for d in dates):
            raise ValueError('Weather table does not cover all accepted AMD dates.')
        selected = [index[d] for d in dates]
        features = {name: np.where(np.isfinite(data), values[selected], np.nan).astype(np.float32)
                    for name, values in daily.items()}
        return self._write('meteorology', features, dates, profile, {
            'processing_parameters': engineering, 'weather_source': str(self._path(table['path'])),
            'aggregation': 'daily calendar windows including cloudy days, sampled on accepted AMD dates',
        })
