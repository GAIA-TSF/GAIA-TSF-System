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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from subsystems.dag.core.interfaces import Pipeline
from subsystems.dag.plugins.features.amd_features import calculate_amd_index
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
        self._points(data, tuple(value.isoformat() for value in dates), self._reference_profile(feature_path), masks, eda_dir, quality)
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
        if not scl_name:
            quality['warnings'].append(
                'No pixel quality filtering configured; only band nodata is excluded.'
            )
        stacks = []
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
                    usable &= np.isfinite(scl) & ~np.isin(
                        scl, filtering['excluded_scl_classes']
                    )
                index = calculate_amd_index(
                    first,
                    second,
                    engineering['method'],
                    float(engineering.get('denominator_epsilon', 0)),
                )
                index[~(domain & usable)] = np.nan
            stacks.append(index)
            quality['acquisitions'].append(
                {
                    'date': acquired.isoformat(),
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
        data = np.stack(stacks)
        dates = tuple(item[0].isoformat() for item in dated)
        quality['gap_days'] = [
            int((b[0] - a[0]).days) for a, b in pairwise(dated)
        ]
        formula = dict(engineering)
        output_dir = self._path(self.options['results']['features']['output_dir'])
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
        _json(
            output_dir / 'metadata.json',
            {
                'acquisition_dates': dates,
                'feature_names': ['amd_index'],
                'formula': formula,
                'source_paths': [str(item[1]) for item in dated],
            },
        )
        if not write_eda:
            return {
                'pipeline': 'amd_index',
                'acquisitions': len(dates),
                'index': str(output_dir / 'amd_index.tif'),
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
        self._points(data, dates, reference, masks, eda_dir, quality)
        return {
            'pipeline': 'amd_eda',
            'acquisitions': len(dates),
            'index': str(output_dir / 'amd_index.tif'),
            'output_dir': str(eda_dir),
        }

    def _points(self, data, dates, reference, masks, output_dir, quality):
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
                if not np.any(counts):
                    quality['warnings'].append(
                        f'No valid observations at {region} point {label}.'
                    )
                ax.plot(
                    [datetime.fromisoformat(day) for day in dates],
                    means,
                    marker='.',
                    color=color,
                    label=f'{region}: {label}',
                )
                records.extend(
                    {
                        'region': region,
                        'point': label,
                        'date': day,
                        'amd_index': float(value) if np.isfinite(value) else None,
                        'valid_pixels': int(count),
                    }
                    for day, value, count in zip(dates, means, counts)
                )
        formula = self.options['feature_engineering']
        operator = '/' if formula['method'] == 'ratio' else '−'
        ax.set(
            title='AMD index at water observation points',
            xlabel='Acquisition date',
            ylabel=f'amd_index ({formula["first_band"]} {operator} {formula["second_band"]})',
        )
        ax.legend()
        ax.grid(alpha=0.25)
        figure.savefig(output_dir / 'point_timeseries.png', dpi=200)
        _csv(output_dir / 'point_timeseries.csv', records)
