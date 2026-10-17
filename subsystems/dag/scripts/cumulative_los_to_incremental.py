"""Convert cumulative LOS rasters to displacement between acquisitions.

Run with no arguments to use the Jagersfontein input/output directories.
Values retain their input units. The first acquisition has no output interval.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import rasterio


PROJECT = Path('/home/lukas/GAIA-TSF/tsf_experiments/slope_monitoring_jagersfontein')
DATE_PATTERN = re.compile(r'(?:tsf_)?los_(\d{8})\.tif$')


def acquisition_date(path):
    match = DATE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f'Expected los_YYYYMMDD.tif or tsf_los_YYYYMMDD.tif: {path}')
    return datetime.strptime(match.group(1), '%Y%m%d').date()


def convert(input_dir, output_dir, pattern='los_*.tif', overwrite=False):
    """Write later-minus-earlier rasters, retaining only jointly valid pixels."""
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError('Input and output directories must differ.')
    dated = sorted((acquisition_date(path), path) for path in input_dir.glob(pattern))
    if len(dated) < 2:
        raise ValueError(f'At least two cumulative rasters are required in {input_dir}.')
    if len({day for day, _ in dated}) != len(dated):
        raise ValueError('Multiple cumulative rasters have the same acquisition date.')
    targets = [output_dir / path.name for _, path in dated[1:]]
    for target in targets:
        if target.resolve() in {path.resolve() for _, path in dated}:
            raise ValueError(f'Output would overwrite a cumulative input: {target}')
        if target.exists() and not overwrite:
            raise FileExistsError(f'Output exists: {target}. Use --overwrite to replace it.')
    # Validate the entire series before creating outputs.
    grid = None
    for _, path in dated:
        with rasterio.open(path) as source:
            if source.count != 1 or source.crs is None:
                raise ValueError(f'Expected a single-band georeferenced raster: {path}')
            current_grid = (source.crs, source.transform, source.width, source.height)
            if grid is not None and current_grid != grid:
                raise ValueError(f'Cumulative raster grids do not match: {path}')
            grid = current_grid
    output_dir.mkdir(parents=True, exist_ok=True)
    for (earlier_day, earlier), (later_day, later), target in zip(dated, dated[1:], targets):
        with rasterio.open(earlier) as previous, rasterio.open(later) as current:
            profile = current.profile.copy()
            profile.update(dtype='float32', nodata=np.nan, count=1)
            with rasterio.open(target, 'w', **profile) as destination:
                for _, window in current.block_windows(1):
                    before = previous.read(1, window=window, masked=True).astype('float64').filled(np.nan)
                    after = current.read(1, window=window, masked=True).astype('float64').filled(np.nan)
                    valid = np.isfinite(before) & np.isfinite(after)
                    values = np.where(valid, after - before, np.nan).astype('float32')
                    destination.write(values, 1, window=window)
                units = current.units[0] or current.tags().get('units')
                if units:
                    destination.set_band_unit(1, units)
                    destination.update_tags(units=units)
                destination.set_band_description(1, 'incremental_los_displacement')
                destination.update_tags(
                    displacement_type='incremental',
                    interval_start=earlier_day.isoformat(), interval_end=later_day.isoformat(),
                    interval_days=(later_day - earlier_day).days,
                    source_previous=earlier.name, source_current=later.name,
                )
        print(f'{target.name}: {earlier_day} -> {later_day}')
    print(f'Wrote {len(targets)} incremental rasters to {output_dir}; first acquisition omitted.')
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'inputs/los_cumulative')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'inputs/los')
    parser.add_argument('--pattern', default='los_*.tif')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output rasters.')
    args = parser.parse_args(argv)
    convert(args.input_dir, args.output_dir, args.pattern, args.overwrite)


if __name__ == '__main__':
    main()
