"""Implement DA_R_03 by spatially co-locating InSAR and in-situ samples."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.windows import Window

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config.yaml'
DATE_PATTERN = re.compile(r'(20\d{6})')


def load_config(path: Path) -> dict[str, Any]:
    """Load a DAG YAML mapping, rejecting non-mapping documents."""
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise TypeError(f'Configuration must be a mapping: {path}')
    return config


def parse_date(filename: str) -> datetime:
    """Extract an acquisition date encoded as YYYYMMDD in a raster filename."""
    match = DATE_PATTERN.search(filename)
    if not match:
        raise ValueError(f'Cannot parse an acquisition date from {filename!r}')
    return datetime.strptime(match.group(1), '%Y%m%d').replace(tzinfo=timezone.utc)


def sample_neighbourhood(
    src: rasterio.io.DatasetReader,
    longitude: float,
    latitude: float,
    window_size: int,
) -> float:
    """Transform a WGS84 location and average its raster neighborhood.

    The result is the nodata-aware mean of an odd square window centered on the
    containing pixel, expressed in the raster's native unit.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            'in_situ.validation.sampling_window_size must be a positive odd integer'
        )
    transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    row, col = src.index(x, y)
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        raise ValueError(
            f'In-situ location ({longitude}, {latitude}) is outside {src.name}'
        )
    radius = window_size // 2
    values = src.read(
        1,
        window=Window(col - radius, row - radius, window_size, window_size),
        boundless=True,
        masked=True,
    )
    return float(values.mean()) if values.count() else float('nan')


def comparison_statistics(comparison: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate JSON-safe statistics for finite paired samples.

    Results include counts, means, InSAR-minus-in-situ bias, MAE, RMSE,
    Pearson correlation, and predictive R². Undefined metrics are ``None``.
    """
    valid = comparison.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        raise ValueError('No finite InSAR/in-situ pairs are available for validation')
    insar = valid['insar_los'].to_numpy(dtype=float)
    insitu = valid['insitu_deformation'].to_numpy(dtype=float)
    residual = insar - insitu
    correlation = (
        float(np.corrcoef(insar, insitu)[0, 1])
        if len(valid) > 1 and np.std(insar) > 0 and np.std(insitu) > 0
        else None
    )
    denominator = float(np.sum((insitu - np.mean(insitu)) ** 2))
    return {
        'sample_count': len(valid),
        'excluded_count': int(len(comparison) - len(valid)),
        'mean_insar_los': float(np.mean(insar)),
        'mean_insitu_deformation': float(np.mean(insitu)),
        'bias': float(np.mean(residual)),
        'mean_absolute_error': float(np.mean(np.abs(residual))),
        'root_mean_squared_error': float(np.sqrt(np.mean(residual**2))),
        'pearson_correlation': correlation,
        'r_squared': (
            float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else None
        ),
    }


def compare(config_path: Path, project_dir: Path | None = None) -> pd.DataFrame:
    """Create independent InSAR/in-situ validation artifacts.

    Rows and rasters are joined on acquisition date. Satellite values are
    neighborhood means around CSV locations. Configured output paths receive a
    two-column comparison CSV and statistical JSON report.
    """
    config = load_config(config_path)
    settings = config.get('in_situ')
    if not isinstance(settings, dict):
        raise TypeError('in_situ must be a mapping in the DAG configuration')
    validation = settings.get('validation')
    if not isinstance(validation, dict):
        raise TypeError('in_situ.validation must be a mapping')
    root = project_dir or config.get('project_dir')
    if not root:
        raise ValueError('project_dir is required')
    project_dir = Path(root).expanduser().resolve()

    insitu_path = project_dir / settings.get(
        'output_csv', 'inputs/in_situ_deformation.csv'
    )
    insitu = pd.read_csv(insitu_path)
    required = {'DATE', 'LATITUDE', 'LONGITUDE', 'LOS_DEFORMATION'}
    missing = required.difference(insitu.columns)
    if missing:
        raise ValueError(f'Missing in-situ CSV columns: {sorted(missing)}')
    insitu['acquisition_date'] = pd.to_datetime(insitu['DATE'], utc=True).dt.date

    insar_dir = project_dir / validation.get('insar_directory', 'inputs/los')
    insar_files = sorted(insar_dir.glob(validation.get('filename_pattern', '*.tif')))
    if not insar_files:
        raise FileNotFoundError(f'No InSAR GeoTIFFs found in {insar_dir}')
    window_size = int(validation.get('sampling_window_size', 3))
    scale = float(validation.get('unit_scale', 1000.0))

    rows: list[dict[str, float]] = []
    for tif in insar_files:
        acquisition_date = parse_date(tif.name).date()
        observations = insitu.loc[insitu['acquisition_date'] == acquisition_date]
        with rasterio.open(tif) as src:
            if src.crs is None:
                raise ValueError(f'InSAR raster has no CRS: {tif}')
            for observation in observations.itertuples(index=False):
                rows.append(
                    {
                        'insar_los': sample_neighbourhood(
                            src,
                            float(observation.LONGITUDE),
                            float(observation.LATITUDE),
                            window_size,
                        )
                        * scale,
                        'insitu_deformation': float(observation.LOS_DEFORMATION),
                    }
                )
    comparison = pd.DataFrame(rows, columns=['insar_los', 'insitu_deformation'])
    if comparison.empty:
        raise ValueError('InSAR and in-situ inputs have no matching acquisition dates')
    statistics = comparison_statistics(comparison)
    csv_path = project_dir / validation.get(
        'output_csv', 'results/validation/insar_insitu_colocation.csv'
    )
    json_path = project_dir / validation.get(
        'statistics_json', 'results/validation/insar_insitu_statistics.json'
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)
    with json_path.open('w', encoding='utf-8') as stream:
        json.dump(statistics, stream, indent=2)
        stream.write('\n')
    return comparison


def main() -> None:
    """Parse command-line options and run the comparison step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--project-dir', type=Path)
    args = parser.parse_args()
    comparison = compare(args.config, args.project_dir)
    print(f'Written {len(comparison)} validation pairs')


if __name__ == '__main__':
    main()
