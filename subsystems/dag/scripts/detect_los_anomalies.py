"""Flag unusual masked LOS acquisitions; write diagnostics without changing rasters.

Scores measure departures from neighboring acquisitions using median/MAD.
Flags identify candidates for review, not proof of processing errors.
For incremental inputs, inspect both interval endpoints in the cumulative series.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re

import numpy as np


PROJECT = Path('/home/lukas/GAIA-TSF/tsf_experiments/slope_monitoring_jagersfontein')
METRICS = ('mean', 'median', 'std', 'p5', 'valid_fraction')


def score_records(records, window=11, threshold=6.0):
    """Compare each date with neighbors excluding itself; use global MAD as fallback."""
    if window < 5 or window % 2 != 1:
        raise ValueError('Window must be an odd number of acquisitions, at least 5.')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be finite and positive.')
    if len(records) < 5:
        raise ValueError('At least five acquisitions are required for anomaly detection.')
    result = [dict(record) for record in records]
    for row in result:
        row['reasons'] = []
    for metric in METRICS:
        values = np.array([row[metric] for row in records], dtype=float)
        finite = values[np.isfinite(values)]
        global_scale = (1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
        for i, value in enumerate(values):
            start, stop = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
            neighbors = values[start:stop]
            neighbors = np.delete(neighbors, i - start)
            neighbors = neighbors[np.isfinite(neighbors)]
            score, baseline = None, None
            if np.isfinite(value) and neighbors.size >= 3:
                baseline = float(np.median(neighbors))
                scale = 1.4826 * np.median(np.abs(neighbors - baseline))
                scale = max(scale, global_scale * 0.25, 1e-6)
                score = float((value - baseline) / scale)
                # Large spread and lost coverage are diagnostic; low spread is not.
                unusual = (score > threshold if metric == 'std' else
                           score < -threshold if metric == 'valid_fraction' else
                           abs(score) > threshold)
                if unusual:
                    result[i]['reasons'].append(metric)
            result[i][f'{metric}_baseline'] = baseline
            result[i][f'{metric}_score'] = score
    for row in result:
        if row['valid_fraction'] == 0:
            row['reasons'].append('no_valid_pixels')
        row['flagged'] = bool(row['reasons'])
        row['reasons'] = ';'.join(row['reasons'])
    return result


def read_records(input_dir, mask_path, pattern):
    import rasterio

    with rasterio.open(mask_path) as source:
        if source.count != 1:
            raise ValueError('TSF mask must be single-band.')
        values = source.read(1, masked=True).astype(float).filled(np.nan)
        mask = np.isfinite(values) & (values != 0)
        grid = (source.crs, source.transform, source.width, source.height)
    if not mask.any():
        raise ValueError('TSF mask contains no selected pixels.')
    dated = []
    for path in input_dir.glob(pattern):
        match = re.fullmatch(r'(?:tsf_)?los_(\d{8})\.tif', path.name)
        if match is None:
            raise ValueError(f'Expected los_YYYYMMDD.tif: {path.name}')
        dated.append((datetime.strptime(match[1], '%Y%m%d').date(), path))
    dated.sort()
    if len({day for day, _ in dated}) != len(dated):
        raise ValueError('Duplicate acquisition dates.')
    records = []
    for day, path in dated:
        with rasterio.open(path) as source:
            if source.count != 1 or grid != (source.crs, source.transform, source.width, source.height):
                raise ValueError(f'LOS raster must be single-band and match the TSF mask grid: {path}')
            data = source.read(1, masked=True).astype(float).filled(np.nan)[mask]
            finite = data[np.isfinite(data)]
            tags = source.tags()
            records.append({
                'date': day.isoformat(), 'filename': path.name,
                'interval_start': tags.get('interval_start', ''),
                'interval_end': tags.get('interval_end', ''),
                'valid_pixels': int(finite.size),
                'valid_fraction': float(finite.size / mask.sum()),
                'mean': float(np.mean(finite)) if finite.size else np.nan,
                'median': float(np.median(finite)) if finite.size else np.nan,
                'std': float(np.std(finite)) if finite.size else np.nan,
                'p5': float(np.percentile(finite, 5)) if finite.size else np.nan,
            })
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'inputs/los')
    parser.add_argument('--mask', type=Path, default=PROJECT / 'static/tsf_mask.tif')
    parser.add_argument('--pattern', default='los_*.tif')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/eda/los_anomalies.csv')
    parser.add_argument('--window', type=int, default=11, help='Local window in acquisitions (default: 11).')
    parser.add_argument('--threshold', type=float, default=6.0, help='Robust score threshold (default: 6).')
    args = parser.parse_args(argv)
    records = score_records(read_records(args.input_dir, args.mask, args.pattern), args.window, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    flagged = [row for row in records if row['flagged']]
    for row in flagged:
        print(f"{row['date']}  {row['filename']}  reasons={row['reasons']}")
    print(f'{len(flagged)} of {len(records)} dates flagged. Report: {args.output}')


if __name__ == '__main__':
    main()
