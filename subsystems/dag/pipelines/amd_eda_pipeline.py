"""Sentinel-2 index engineering followed by regional and point AMD EDA."""

from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from scipy.ndimage import binary_dilation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from subsystems.dag.core.interfaces import Pipeline
from subsystems.dag.plugins.features.amd_features import calculate_amd_index
from subsystems.dag.plugins.eda.amd_gaps import analyse_amd_gaps
from subsystems.dag.plugins.eda.amd_noise import clean_water_variability, spatial_inconsistency
from subsystems.dag.plugins.eda.amd_trends import (
    fill_short_gaps, process_trend_stack, robust_lowess,
)
from subsystems.dag.utils.raster import RasterProfile, write_raster
from subsystems.dag.utils.statistics import feature_statistics


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def _csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AMDEDAPipeline(Pipeline):
    """Generate an AMD index before analysing both configured water regions."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        root = Path(self.config['project_dir']).expanduser()
        self.root = root if root.is_absolute() else self.config_path.parent / root
        self.options = self.config['amd']

    def _path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _check_grid(source, reference):
        if source.crs is None or (
            source.crs,
            source.transform,
            source.height,
            source.width,
        ) != (
            reference['crs'],
            reference['transform'],
            reference['height'],
            reference['width'],
        ):
            raise ValueError(f'AMD grid mismatch or missing CRS: {source.name}')

    @staticmethod
    def _date(path, metadata):
        match = re.search(r'_MSIL2A_(\d{8}T\d{6})_', path.name)
        if not match:
            raise ValueError(f'Invalid Sentinel-2 acquisition filename: {path.name}')
        acquired = date.fromisoformat(match[1][:8])
        timestamp = metadata.get('properties', {}).get('datetime') or metadata.get(
            'PRODUCT_START_TIME'
        )
        if (
            timestamp
            and datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
            != acquired
        ):
            raise ValueError(f'Acquisition date differs from metadata: {path.name}')
        return acquired

    @staticmethod
    def _band(source, name, mapping, window=None):
        position = mapping.get(name)
        if position is None:
            matches = [
                i + 1 for i, value in enumerate(source.descriptions) if value == name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f'Set band_positions.{name}; raster band mapping is ambiguous.'
                )
            position = matches[0]
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 1 <= position <= source.count
        ):
            raise ValueError(f'Invalid raster band position for {name}: {position}')
        description = source.descriptions[position - 1]
        if description and description != name:
            raise ValueError(
                f'Band description {description} conflicts with configured {name}.'
            )
        return source.read(position, masked=True, window=window).astype(float).filled(np.nan)

    @staticmethod
    def _input_window(source, reference):
        """Crop a larger aligned image to the mask grid without resampling."""
        transform = reference['transform']
        if source.crs != reference['crs'] or source.crs is None or (
            source.transform.a, source.transform.b, source.transform.d, source.transform.e
        ) != (transform.a, transform.b, transform.d, transform.e):
            raise ValueError(f'AMD grid mismatch: {source.name}')
        col, row = ~source.transform * (transform.c, transform.f)
        if not np.allclose([col, row], np.round([col, row]), rtol=0, atol=1e-6):
            raise ValueError(f'AMD grid mismatch: unaligned origin in {source.name}')
        col, row = round(col), round(row)
        if col < 0 or row < 0 or col + reference['width'] > source.width or row + reference['height'] > source.height:
            raise ValueError(f'AMD image does not cover the mask grid: {source.name}')
        return rasterio.windows.Window(col, row, reference['width'], reference['height'])

    def run(self):
        result = self._run_index_generation()
        return self._run_eda_from_generation(result)

    def _run_index_generation(self):
        inputs = self.options['inputs']['sentinel2']
        engineering = self.options['feature_engineering']
        first_name, second_name = engineering['first_band'], engineering['second_band']
        if not first_name or not second_name:
            raise ValueError('Configure AMD first_band and second_band before running.')
        method = engineering['method']
        epsilon = float(engineering.get('denominator_epsilon', 0))
        # Validate the formula even if no input files are present.
        calculate_amd_index(np.array([1.0]), np.array([1.0]), method, epsilon)
        scale = float(engineering.get('scale', 1))
        offset = float(engineering.get('offset', 0))
        if not np.isfinite(scale) or scale == 0 or not np.isfinite(offset):
            raise ValueError(
                'AMD scale must be finite and nonzero; offset must be finite.'
            )
        directory = self._path(inputs['directory'])
        paths = sorted(directory.glob(inputs.get('filename_pattern', '*.jp2')))
        if not paths:
            raise FileNotFoundError(f'No Sentinel-2 images found in {directory}')
        eda_dir = self._path(self.options['results']['eda']['output_dir'])
        eda_dir.mkdir(parents=True, exist_ok=True)
        quality = {'errors': [], 'warnings': [], 'acquisitions': []}
        inventory, dated = [], []
        for path in paths:
            sidecar = path.with_suffix('.json')
            row = {
                'image': str(path),
                'metadata': str(sidecar),
                'date': '',
                'status': 'ok',
            }
            try:
                metadata = json.loads(sidecar.read_text())
                acquired = self._date(path, metadata)
                row['date'] = acquired.isoformat()
                with rasterio.open(path) as source:
                    row.update(
                        bands=source.count,
                        width=source.width,
                        height=source.height,
                        crs=str(source.crs),
                        resolution_x=source.res[0],
                        resolution_y=source.res[1],
                    )
                dated.append((acquired, path, metadata))
            except (OSError, ValueError, TypeError) as exc:
                row['status'] = 'error'
                quality['errors'].append(f'{path.name}: {exc}')
            inventory.append(row)
        # Keep a stable schema for failed inventory entries too.
        fields = dict.fromkeys(key for row in inventory for key in row)
        inventory = [{key: row.get(key, '') for key in fields} for row in inventory]
        known = {p.with_suffix('.json') for p in paths}
        quality['unpaired_metadata'] = [
            str(p) for p in sorted(directory.glob('*.json')) if p not in known
        ]
        start = date.fromisoformat(inputs['start_date']) if inputs.get('start_date') else None
        end = date.fromisoformat(inputs['end_date']) if inputs.get('end_date') else None
        dated = [item for item in dated if (start is None or item[0] >= start) and (end is None or item[0] <= end)]
        unique = {}
        for item in sorted(dated, key=lambda value: (value[0], value[1].name)):
            if item[0] in unique:
                quality.setdefault('duplicate_acquisitions', {}).setdefault(item[0].isoformat(), []).append(str(item[1]))
            else:
                unique[item[0]] = item
        dated = list(unique.values())
        _csv(eda_dir / 'inventory.csv', inventory)
        try:
            if quality['errors']:
                raise ValueError(
                    'AMD inventory validation failed: '
                    + '; '.join(quality['errors'])
                )
            result = self._process(sorted(dated), engineering, inputs, quality, eda_dir, write_eda=False)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            if str(exc) not in quality['errors']:
                quality['errors'].append(str(exc))
            raise
        finally:
            _json(eda_dir / 'quality_report.json', quality)
        return result

    def _run_eda_from_generation(self, result):
        """Run EDA after the index-generation stage has completed."""
        # The generation method already performs the shared validation and
        # writes the index. Re-read its output so EDA is a genuine consumer.
        feature_path = Path(result['index'])
        with rasterio.open(feature_path) as source:
            data = source.read().astype(float)
            dates = tuple(date.fromisoformat(value) for value in source.descriptions)
        filtered_path = Path(result['cloud_edge_filtered_index'])
        with rasterio.open(filtered_path) as source:
            filtered_data = source.read().astype(float)
        masks = {}
        for region in ('amd', 'clean_water'):
            with rasterio.open(self._path(self.options['static'][f'{region}_mask'])) as source:
                masks[region] = source.read(1).astype(bool)
        eda_dir = self._path(self.options['results']['eda']['output_dir'])
        statistics = {'feature': 'amd_index', 'formula': self.options['feature_engineering'], 'regions': {}}
        for region, mask in masks.items():
            statistics['regions'][region] = {
                'overall': feature_statistics(data[:, mask]),
                'per_acquisition': {day.isoformat(): feature_statistics(data[i, mask]) for i, day in enumerate(dates)},
            }
        _json(eda_dir / 'statistics.json', statistics)
        quality = {}
        date_strings = tuple(value.isoformat() for value in dates)
        variability_data, variability = self._clean_water_noise(
            filtered_data, date_strings, masks['clean_water'], eda_dir
        )
        profile = self._reference_profile(feature_path)
        spatial_data, spatial = self._spatial_noise(
            variability_data, date_strings, masks['clean_water'], eda_dir, profile
        )
        point_records = self._points(data, date_strings, profile, masks, eda_dir,
                                     quality, filtered_data, variability_data, spatial_data)
        self._trends(spatial_data, dates, masks, profile, point_records, eda_dir)
        self._gaps(data, date_strings, masks, point_records, eda_dir)
        return {**result, 'pipeline': 'amd_eda'}

    @staticmethod
    def _reference_profile(path):
        with rasterio.open(path) as source:
            return source.profile

    def _process(self, dated, engineering, inputs, quality, eda_dir, write_eda=True):
        with rasterio.open(self._path(self.options['static']['amd_mask'])) as source:
            reference = source.profile
            self._check_grid(source, reference)
        masks = {}
        for region in ('amd', 'clean_water'):
            with rasterio.open(
                self._path(self.options['static'][f'{region}_mask'])
            ) as source:
                self._check_grid(source, reference)
                if source.count != 1:
                    raise ValueError('AMD region masks must be single-band.')
                values = source.read(1, masked=True).filled(0)
                if not np.all(np.isin(values, [0, 1])) or not np.any(values == 1):
                    raise ValueError(f'{region} mask must be binary and nonempty.')
                masks[region] = values == 1
        quality['mask_overlap_pixels'] = int(
            np.count_nonzero(masks['amd'] & masks['clean_water'])
        )
        if quality['mask_overlap_pixels']:
            quality['warnings'].append('AMD and clean-water masks overlap.')
        domain = masks['amd'] | masks['clean_water']
        mapping = inputs.get('band_positions', {})
        filtering = self.options.get('quality', {})
        scl_name = filtering.get('scl_band')
        acquisition_filter = filtering.get('acquisition_cloud_filter', {})
        if not isinstance(acquisition_filter, dict):
            raise TypeError('quality.acquisition_cloud_filter must be a mapping.')
        acquisition_filter_enabled = bool(acquisition_filter.get('enabled', False))
        maximum_cloudy_pixels = acquisition_filter.get('maximum_cloudy_pixels', 0)
        if (isinstance(maximum_cloudy_pixels, bool)
                or not isinstance(maximum_cloudy_pixels, int)
                or maximum_cloudy_pixels < 0):
            raise ValueError('maximum_cloudy_pixels must be a non-negative integer.')
        cloudy_classes = acquisition_filter.get('cloudy_scl_classes', [3, 8, 9, 10])
        if not isinstance(cloudy_classes, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in cloudy_classes
        ):
            raise ValueError('cloudy_scl_classes must be a list of integer SCL classes.')
        if acquisition_filter_enabled and not scl_name:
            raise ValueError('Acquisition cloud filtering requires quality.scl_band.')
        if not scl_name:
            quality['warnings'].append(
                'No pixel quality filtering configured; only band nodata is excluded.'
            )
        stacks = []
        filtered_stacks = []
        accepted_dated = []
        buffer_pixels = filtering.get('cloud_edge_buffer_pixels', 0)
        if isinstance(buffer_pixels, bool) or not isinstance(buffer_pixels, int) or buffer_pixels < 0:
            raise ValueError('quality.cloud_edge_buffer_pixels must be a non-negative integer.')
        for acquired, path, metadata in dated:
            with rasterio.open(path) as source:
                window = self._input_window(source, reference)
                first = self._band(source, engineering['first_band'], mapping, window)
                second = self._band(source, engineering['second_band'], mapping, window)
                scale, offset = (
                    float(engineering.get('scale', 1)),
                    float(engineering.get('offset', 0)),
                )
                first, second = first * scale + offset, second * scale + offset
                usable = np.isfinite(first) & np.isfinite(second)
                if scl_name:
                    scl = self._band(source, scl_name, mapping, window)
                    excluded = ~np.isfinite(scl) | np.isin(
                        scl, filtering['excluded_scl_classes'])
                    usable &= ~excluded
                    cloudy_pixels = int(np.count_nonzero(
                        domain & np.isin(scl, cloudy_classes)
                    ))
                else:
                    cloudy_pixels = 0
                index = calculate_amd_index(
                    first,
                    second,
                    engineering['method'],
                    float(engineering.get('denominator_epsilon', 0)),
                )
                index[~(domain & usable)] = np.nan
                filtered_index = index.copy()
                if scl_name and buffer_pixels:
                    buffered = binary_dilation(excluded, iterations=buffer_pixels)
                    filtered_index[buffered] = np.nan
            accepted = (
                not acquisition_filter_enabled
                or cloudy_pixels <= maximum_cloudy_pixels
            )
            if accepted:
                stacks.append(index)
                filtered_stacks.append(filtered_index)
                accepted_dated.append((acquired, path, metadata))
            quality['acquisitions'].append(
                {
                    'date': acquired.isoformat(),
                    'cloudy_pixels_in_analysis_masks': cloudy_pixels,
                    'accepted_by_acquisition_cloud_filter': accepted,
                    'scene_cloud_cover_pct': metadata.get(
                        'cloud_cover_pct',
                        metadata.get('properties', {}).get('eo:cloud_cover'),
                    ),
                    'regions': {
                        region: {
                            'pixels': int(mask.sum()),
                            'valid_pixels': int(np.isfinite(index[mask]).sum()),
                            'valid_fraction': float(np.isfinite(index[mask]).mean()),
                        }
                        for region, mask in masks.items()
                    },
                }
            )
        quality['acquisition_cloud_filter'] = {
            'enabled': acquisition_filter_enabled,
            'maximum_cloudy_pixels': maximum_cloudy_pixels,
            'cloudy_scl_classes': cloudy_classes,
            'accepted_acquisitions': len(accepted_dated),
            'rejected_acquisitions': len(dated) - len(accepted_dated),
            'rejected_dates': [
                row['date'] for row in quality['acquisitions']
                if not row['accepted_by_acquisition_cloud_filter']
            ],
        }
        if not accepted_dated:
            raise ValueError('Acquisition cloud filter rejected every Sentinel-2 image.')
        data = np.stack(stacks)
        filtered_data = np.stack(filtered_stacks)
        dates = tuple(item[0].isoformat() for item in accepted_dated)
        quality['gap_days'] = [
            int((b[0] - a[0]).days) for a, b in pairwise(accepted_dated)
        ]
        formula = dict(engineering)
        output_options = self.options['results'].get('index', self.options['results']['features'])
        output_dir = self._path(output_options['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        profile = RasterProfile(
            reference['crs'],
            reference['transform'],
            reference['width'],
            reference['height'],
            'float32',
            np.nan,
        )
        write_raster(output_dir / 'amd_index.tif', data, profile, 'GTiff', dates)
        filtered_path = output_dir / 'amd_index_cloud_edge_filtered.tif'
        write_raster(filtered_path, filtered_data, profile, 'GTiff', dates)
        variability_data, variability = self._clean_water_noise(
            filtered_data, dates, masks['clean_water'], eda_dir
        )
        spatial_data, spatial = self._spatial_noise(
            variability_data, dates, masks['clean_water'], eda_dir, reference
        )
        _json(
            output_dir / 'metadata.json',
            {
                'acquisition_dates': dates,
                'feature_names': ['amd_index'],
                'formula': formula,
                'source_paths': [str(item[1]) for item in accepted_dated],
                'cloud_edge_filtered_index': str(filtered_path),
                'cloud_edge_buffer_pixels': buffer_pixels,
            },
        )
        if not write_eda:
            return {
                'pipeline': 'amd_index',
                'acquisitions': len(dates),
                'index': str(output_dir / 'amd_index.tif'),
                'cloud_edge_filtered_index': str(filtered_path),
                'output_dir': str(eda_dir),
            }
        statistics = {'feature': 'amd_index', 'formula': formula, 'regions': {}}
        for region, mask in masks.items():
            statistics['regions'][region] = {
                'overall': feature_statistics(data[:, mask]),
                'per_acquisition': {
                    day: feature_statistics(data[i, mask])
                    for i, day in enumerate(dates)
                },
            }
        _json(eda_dir / 'statistics.json', statistics)
        point_records = self._points(data, dates, reference, masks, eda_dir, quality,
                                     filtered_data, variability_data, spatial_data)
        self._trends(
            spatial_data, tuple(date.fromisoformat(value) for value in dates),
            masks, reference, point_records, eda_dir,
        )
        self._gaps(data, dates, masks, point_records, eda_dir)
        return {
            'pipeline': 'amd_eda',
            'acquisitions': len(dates),
            'index': str(output_dir / 'amd_index.tif'),
            'cloud_edge_filtered_index': str(filtered_path),
            'output_dir': str(eda_dir),
        }

    def _points(self, data, dates, reference, masks, output_dir, quality,
                filtered_data=None, variability_data=None, spatial_data=None):
        options = self.options.get('points', {})
        window = options.get('window_size', 1)
        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or window < 1
            or window % 2 == 0
        ):
            raise ValueError('Point window_size must be a positive odd integer.')
        radius = window // 2
        figure = Figure(figsize=(12, 5), layout='constrained')
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        records = []
        for region, color in (('amd', 'tab:orange'), ('clean_water', 'tab:blue')):
            points = gpd.read_file(
                self._path(self.options['static'][f'{region}_point'])
            )
            if points.crs is None or points.empty:
                raise ValueError(f'{region} points must be nonempty and have a CRS.')
            points = points.to_crs(reference['crs'])
            for number, (_, point) in enumerate(points.iterrows(), 1):
                geometry = point.geometry
                if (
                    geometry is None
                    or geometry.is_empty
                    or geometry.geom_type != 'Point'
                ):
                    raise ValueError(f'{region} geometry must be a nonempty Point.')
                row, col = rasterio.transform.rowcol(
                    reference['transform'], geometry.x, geometry.y
                )
                if (
                    not (0 <= row < data.shape[1] and 0 <= col < data.shape[2])
                    or not masks[region][row, col]
                ):
                    raise ValueError(
                        f'{region} point {number} is outside its region mask.'
                    )
                label = str(point.get(options.get('label_column', 'label'), number))
                rows = slice(max(0, row - radius), min(data.shape[1], row + radius + 1))
                cols = slice(max(0, col - radius), min(data.shape[2], col + radius + 1))
                values = np.where(
                    masks[region][rows, cols], data[:, rows, cols], np.nan
                )
                counts = np.isfinite(values).sum(axis=(1, 2))
                means = np.divide(
                    np.nansum(values, axis=(1, 2)),
                    counts,
                    out=np.full(len(dates), np.nan),
                    where=counts > 0,
                )
                filtered_means = means.copy()
                if filtered_data is not None:
                    filtered_values = np.where(
                        masks[region][rows, cols],
                        filtered_data[:, rows, cols], np.nan
                    )
                    filtered_counts = np.isfinite(filtered_values).sum(axis=(1, 2))
                    filtered_means = np.divide(
                        np.nansum(filtered_values, axis=(1, 2)), filtered_counts,
                        out=np.full(len(dates), np.nan), where=filtered_counts > 0,
                    )
                variability_means = filtered_means.copy()
                if variability_data is not None:
                    variability_values = np.where(
                        masks[region][rows, cols],
                        variability_data[:, rows, cols], np.nan
                    )
                    variability_counts = np.isfinite(variability_values).sum(axis=(1, 2))
                    variability_means = np.divide(
                        np.nansum(variability_values, axis=(1, 2)), variability_counts,
                        out=np.full(len(dates), np.nan), where=variability_counts > 0,
                    )
                spatial_means = variability_means.copy()
                if spatial_data is not None:
                    spatial_values = np.where(
                        masks[region][rows, cols], spatial_data[:, rows, cols], np.nan
                    )
                    spatial_counts = np.isfinite(spatial_values).sum(axis=(1, 2))
                    spatial_means = np.divide(
                        np.nansum(spatial_values, axis=(1, 2)), spatial_counts,
                        out=np.full(len(dates), np.nan), where=spatial_counts > 0,
                    )
                if not np.any(counts):
                    quality['warnings'].append(
                        f'No valid observations at {region} point {label}.'
                    )
                plot_dates = [datetime.fromisoformat(day) for day in dates]
                years = sorted({day.year for day in plot_dates})
                labelled = False
                for year in years:
                    valid = [
                        i for i, day in enumerate(plot_dates)
                        if day.year == year and np.isfinite(means[i])
                    ]
                    if not valid:
                        continue
                    ax.plot(
                        [plot_dates[i] for i in valid],
                        means[valid],
                        marker='.',
                        color=color,
                        label=f'{region}: {label}' if not labelled else '_nolegend_',
                    )
                    labelled = True
                filtered_labelled = False
                if filtered_data is not None:
                    for year in years:
                        filtered_valid = [
                            i for i, day in enumerate(plot_dates)
                            if day.year == year and np.isfinite(filtered_means[i])
                        ]
                        if not filtered_valid:
                            continue
                        ax.plot(
                            [plot_dates[i] for i in filtered_valid],
                            filtered_means[filtered_valid], marker='.', linestyle='--',
                            color=color, alpha=0.8,
                            label=(f'{region}: {label} (cloud-edge filtered)'
                                   if not filtered_labelled else '_nolegend_'),
                        )
                        filtered_labelled = True
                variability_labelled = False
                if variability_data is not None:
                    for year in years:
                        valid_variability = [
                            i for i, day in enumerate(plot_dates)
                            if day.year == year and np.isfinite(variability_means[i])
                        ]
                        if not valid_variability:
                            continue
                        ax.plot(
                            [plot_dates[i] for i in valid_variability],
                            variability_means[valid_variability], marker='.',
                            linestyle=':', color=color, alpha=0.9,
                            label=(f'{region}: {label} (+ clean-water variability)'
                                   if not variability_labelled else '_nolegend_'),
                        )
                        variability_labelled = True
                spatial_labelled = False
                if spatial_data is not None:
                    for year in years:
                        spatial_valid = [
                            i for i, day in enumerate(plot_dates)
                            if day.year == year and np.isfinite(spatial_means[i])
                        ]
                        if not spatial_valid:
                            continue
                        ax.plot(
                            [plot_dates[i] for i in spatial_valid],
                            spatial_means[spatial_valid], marker='.', linestyle='-.',
                            color=color, alpha=0.9,
                            label=(f'{region}: {label} (+ spatial consistency)'
                                   if not spatial_labelled else '_nolegend_'),
                        )
                        spatial_labelled = True
                records.extend(
                    {
                        'region': region,
                        'point': label,
                        'date': day,
                        'amd_index': float(value) if np.isfinite(value) else None,
                        'cloud_edge_filtered': (
                            float(filtered_value) if np.isfinite(filtered_value) else None
                        ),
                        'cloud_edge_removed': bool(
                            np.isfinite(value) and not np.isfinite(filtered_value)
                        ),
                        'clean_water_variability_filtered': (
                            float(variability_value)
                            if np.isfinite(variability_value) else None
                        ),
                        'clean_water_variability_removed': bool(
                            np.isfinite(filtered_value)
                            and not np.isfinite(variability_value)
                        ),
                        'spatial_consistency_filtered': (
                            float(spatial_value) if np.isfinite(spatial_value) else None
                        ),
                        'spatial_inconsistency_removed': bool(
                            np.isfinite(variability_value) and not np.isfinite(spatial_value)
                        ),
                        'valid_pixels': int(count),
                    }
                    for day, value, filtered_value, variability_value, spatial_value, count in zip(
                        dates, means, filtered_means, variability_means, spatial_means, counts
                    )
                )
        formula = self.options['feature_engineering']
        operator = '/' if formula['method'] == 'ratio' else '−'
        plot_name = 'AMD_difference' if formula['method'] == 'difference' else 'AMD_index'
        ax.set(
            title=f'{plot_name} at water observation points',
            xlabel='Acquisition date',
            ylabel=f'{plot_name} ({formula["first_band"]} {operator} {formula["second_band"]})',
        )
        ax.legend()
        ax.grid(alpha=0.25)
        figure.savefig(output_dir / 'point_timeseries.png', dpi=200)
        _csv(output_dir / 'point_timeseries.csv', records)
        return records

    def _clean_water_noise(self, data, dates, clean_mask, output_dir):
        options = self.options.get('quality', {}).get('clean_water_variability', {})
        if not options.get('enabled', False):
            return data.copy(), None
        result = clean_water_variability(
            data, clean_mask,
            float(options.get('threshold_sigma', 3.0)),
            int(options.get('minimum_valid_pixels', 10)),
        )
        filtered = data.copy()
        filtered[result['noisy']] = np.nan
        rows = [{
            'date': day,
            'clean_water_valid_pixels': int(result['valid_pixels'][i]),
            'clean_water_robust_sigma': (
                float(result['robust_sigma'][i])
                if np.isfinite(result['robust_sigma'][i]) else None
            ),
            'noise_threshold': result['threshold'],
            'noisy_acquisition': bool(result['noisy'][i]),
        } for i, day in enumerate(dates)]
        _csv(output_dir / 'clean_water_variability.csv', rows)
        _json(output_dir / 'clean_water_variability.json', {
            'method': '1.4826 * spatial MAD within clean-water mask per date',
            'threshold_sigma': float(options.get('threshold_sigma', 3.0)),
            'minimum_valid_pixels': int(options.get('minimum_valid_pixels', 10)),
            'baseline_median_sigma': result['baseline_median_sigma'],
            'between_acquisition_robust_sigma': result['between_acquisition_robust_sigma'],
            'noise_threshold': result['threshold'],
            'noisy_acquisitions': [
                day for day, noisy in zip(dates, result['noisy']) if noisy
            ],
        })
        return filtered, result

    def _spatial_noise(self, data, dates, clean_mask, output_dir, reference):
        options = self.options.get('quality', {}).get('spatial_inconsistency', {})
        if not options.get('enabled', False):
            return data.copy(), None
        result = spatial_inconsistency(
            data, clean_mask,
            int(options.get('window_size', 3)),
            float(options.get('threshold_sigma', 3.0)),
            int(options.get('minimum_valid_neighbors', 3)),
            int(options.get('minimum_reference_pixels', 10)),
        )
        filtered = data.copy()
        filtered[result['flags']] = np.nan
        _csv(output_dir / 'spatial_inconsistency.csv', [{
            'date': day,
            'reference_robust_sigma': (
                float(result['robust_sigma'][i])
                if np.isfinite(result['robust_sigma'][i]) else None
            ),
            'residual_threshold': (
                float(result['thresholds'][i])
                if np.isfinite(result['thresholds'][i]) else None
            ),
            'flagged_pixels': int(result['flags'][i].sum()),
        } for i, day in enumerate(dates)])
        frequency = result['flags'].sum(axis=0).astype(np.float32)
        profile = RasterProfile(reference['crs'], reference['transform'],
                                reference['width'], reference['height'],
                                'float32', np.nan)
        write_raster(output_dir / 'spatial_inconsistency_frequency.tif',
                     frequency, profile, 'GTiff')
        _json(output_dir / 'spatial_inconsistency.json', {
            'method': 'absolute local-median residual calibrated per date on clean water',
            'window_size': int(options.get('window_size', 3)),
            'threshold_sigma': float(options.get('threshold_sigma', 3.0)),
            'minimum_valid_neighbors': int(options.get('minimum_valid_neighbors', 3)),
            'minimum_reference_pixels': int(options.get('minimum_reference_pixels', 10)),
            'total_flagged_pixel_observations': int(result['flags'].sum()),
            'dates_with_flags': sum(bool(value) for value in result['flags'].any(axis=(1, 2))),
        })
        return filtered, result

    def _trends(self, data, dates, masks, reference, point_records, eda_dir):
        config = self.options.get('trend_analysis', {})
        if not config.get('enabled', False):
            return
        if config.get('method', 'robust_lowess') != 'robust_lowess':
            raise ValueError('AMD trend method must be robust_lowess.')
        domain = masks['amd'] | masks['clean_water']
        filled, smoothed, interpolation, gap_days = process_trend_stack(
            data, dates, domain, config
        )
        result_config = self.options['results']['trend_features']
        output_dir = self._path(result_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        profile = RasterProfile(reference['crs'], reference['transform'],
                                reference['width'], reference['height'],
                                'float32', np.nan)
        band_names = tuple(value.isoformat() for value in dates)
        outputs = {}
        for name, values in (
            ('amd_filled', filled), ('amd_smoothed', smoothed),
            ('interpolation_mask', interpolation),
            ('interpolation_gap_days', gap_days),
        ):
            path = output_dir / f'{name}.tif'
            write_raster(path, values, profile, 'GTiff', band_names)
            outputs[name] = str(path)

        grouped = {}
        for row in point_records:
            grouped.setdefault((row['region'], row['point']), []).append(row)
        plot = Figure(figsize=(12, 5), layout='constrained')
        FigureCanvasAgg(plot)
        ax = plot.subplots()
        trend_rows = []
        summary = {}
        for (region, label), rows in grouped.items():
            values = np.asarray([
                row['spatial_consistency_filtered']
                if row['spatial_consistency_filtered'] is not None else np.nan
                for row in rows
            ])
            series, flags, gaps = fill_short_gaps(
                values, dates, int(config.get('max_gap_days', 30)),
                bool(config.get('require_same_year', True)),
            )
            trend = robust_lowess(
                series, dates, flags, int(config.get('window_days', 45)),
                int(config.get('robust_iterations', 2)),
                float(config.get('interpolated_weight', .5)),
                bool(config.get('process_each_year', True)),
            )
            color = 'tab:orange' if region == 'amd' else 'tab:blue'
            observed = np.isfinite(values)
            ax.scatter(np.asarray(dates)[observed], values[observed], s=13,
                       color=color, alpha=.55, label=f'{region}: {label} observed')
            ax.scatter(np.asarray(dates)[flags], series[flags], s=28,
                       facecolors='none', edgecolors=color,
                       label=f'{region}: {label} interpolated')
            for year in sorted({value.year for value in dates}):
                selected = np.asarray([value.year == year for value in dates])
                ax.plot(np.asarray(dates)[selected], trend[selected], color=color,
                        linewidth=2.2,
                        label=f'{region}: {label} LOWESS' if year == dates[0].year else '_nolegend_')
            summary[f'{region}:{label}'] = {
                'observed': int(observed.sum()),
                'interpolated': int(flags.sum()),
                'smoothed': int(np.isfinite(trend).sum()),
            }
            trend_rows.extend({
                'region': region, 'point': label, 'date': day.isoformat(),
                'noise_filtered_value': float(raw) if np.isfinite(raw) else None,
                'filled_value': float(value) if np.isfinite(value) else None,
                'smoothed_value': float(smooth) if np.isfinite(smooth) else None,
                'is_observed': bool(np.isfinite(raw)),
                'is_interpolated': bool(flag),
                'gap_length_days': float(gap) if np.isfinite(gap) else None,
            } for day, raw, value, smooth, flag, gap in
                zip(dates, values, series, trend, flags, gaps))
        ax.set(title='Noise-filtered AMD observations and retrospective trends',
               xlabel='Acquisition date', ylabel='AMD_difference (B04 − B02)')
        ax.grid(alpha=.25)
        ax.legend()
        plot.savefig(eda_dir / 'point_trends.png', dpi=200)
        _csv(eda_dir / 'point_trends.csv', trend_rows)
        report = {
            'method': 'bounded linear interpolation followed by robust date-aware LOWESS',
            'parameters': config, 'points': summary,
            'raster_interpolated_pixel_observations': int(interpolation.sum()),
            'output_files': outputs,
            'retrospective_only': True,
        }
        _json(eda_dir / 'gap_filling_report.json', {
            'max_gap_days': int(config.get('max_gap_days', 30)),
            'require_same_year': bool(config.get('require_same_year', True)),
            'raster_interpolated_pixel_observations': int(interpolation.sum()),
            'points': {key: {'interpolated': value['interpolated']}
                       for key, value in summary.items()},
        })
        _json(eda_dir / 'smoothing_report.json', report)
        _json(output_dir / 'metadata.json', {
            'feature_names': list(outputs), 'acquisition_dates': list(band_names),
            'formula': self.options['feature_engineering'], 'processing_parameters': config,
            'input': 'AMD observations after all EDA noise filters',
            'output_files': outputs, 'retrospective_only': True,
        })

    def _gaps(self, data, dates, masks, point_records, output_dir):
        """Write acquisition, regional, and observation-point gap diagnostics."""
        series = {
            'model_domain': np.any(
                np.isfinite(data[:, masks['amd'] | masks['clean_water']]), axis=1
            ),
        }
        series.update({
            f'region:{region}': np.any(np.isfinite(data[:, mask]), axis=1)
            for region, mask in masks.items()
        })
        point_values = {}
        for row in point_records:
            key = f'point:{row["region"]}:{row["point"]}'
            point_values.setdefault(key, []).append(row['amd_index'] is not None)
            filtered_key = f'point_cloud_edge_filtered:{row["region"]}:{row["point"]}'
            point_values.setdefault(filtered_key, []).append(
                row['cloud_edge_filtered'] is not None
            )
            variability_key = (
                f'point_clean_water_variability_filtered:'
                f'{row["region"]}:{row["point"]}'
            )
            point_values.setdefault(variability_key, []).append(
                row['clean_water_variability_filtered'] is not None
            )
            spatial_key = f'point_spatially_filtered:{row["region"]}:{row["point"]}'
            point_values.setdefault(spatial_key, []).append(
                row['spatial_consistency_filtered'] is not None
            )
        series.update(point_values)
        report, intervals = analyse_amd_gaps(dates, series)
        report['cloud_edge_contamination'] = {
            'buffer_pixels': self.options.get('quality', {}).get(
                'cloud_edge_buffer_pixels', 0
            ),
            'point_observations_removed': sum(
                row['cloud_edge_removed'] for row in point_records
            ),
            'method': 'binary dilation of excluded SCL pixels',
        }
        report['clean_water_variability'] = {
            'point_observations_removed': sum(
                row['clean_water_variability_removed'] for row in point_records
            ),
            'method': 'dates above robust clean-water spatial-variability threshold',
        }
        report['spatial_inconsistency'] = {
            'point_observations_removed': sum(
                row['spatial_inconsistency_removed'] for row in point_records
            ),
            'method': 'local-median residual threshold calibrated on clean water',
        }
        _json(output_dir / 'gap_analysis.json', report)
        _csv(output_dir / 'gap_intervals.csv', intervals)

        figure = Figure(figsize=(12, 5), layout='constrained')
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        years = sorted({date.fromisoformat(value).year for value in dates})
        labels, values = [], []
        for scope, result in report['series'].items():
            for year in years:
                labels.append((scope, year))
                values.append(result['yearly'][str(year)]['valid_fraction'])
        matrix = np.asarray(values).reshape(len(report['series']), len(years))
        image = ax.imshow(matrix, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(years)), years)
        ax.set_yticks(range(len(report['series'])), report['series'])
        ax.set(xlabel='Year', title='AMD valid-observation fraction')
        figure.colorbar(image, ax=ax, label='Valid fraction')
        figure.savefig(output_dir / 'gap_coverage.png', dpi=200)
