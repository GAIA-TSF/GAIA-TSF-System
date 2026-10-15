"""Reproject WGS-84 LOS rasters onto a reference-derived grid.

Example::

    python subsystems/dag/scripts/resample_los_2_utm.py \
        --input-dir /site/sentinel1/results/pre_failure/los \
        --output-dir /site/inputs/los --target-epsg 32735 \
        --reference-raster /site/static/tsf_mask.tif

If the reference already uses the target CRS, its grid is matched exactly.
Otherwise its bounds and dimensions determine a suggested reprojected grid.
Displacement values retain their original units (no mm-to-m conversion).
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

NODATA = -9999.0


def target_grid(reference_path: Path, target_crs: CRS):
    """Read extent/resolution from a reference; preserve exact grid when possible."""
    with rasterio.open(reference_path) as reference:
        if reference.crs is None:
            raise ValueError(f'Reference raster has no CRS: {reference_path}')
        if reference.crs == target_crs:
            return reference.transform, reference.width, reference.height
        return calculate_default_transform(
            reference.crs, target_crs, reference.width, reference.height,
            *reference.bounds,
        )


def reproject_raster(src_path: Path, dst_path: Path, target_crs: CRS, grid) -> None:
    """Reproject one single-band WGS-84 LOS raster using bilinear interpolation."""
    if src_path.resolve() == dst_path.resolve():
        raise ValueError('Input and output raster paths must differ.')
    transform, width, height = grid
    with rasterio.open(src_path) as src:
        if src.crs != CRS.from_epsg(4326):
            raise ValueError(f'Expected WGS-84 (EPSG:4326) input: {src_path}; got {src.crs}')
        if src.count != 1:
            raise ValueError(f'Expected a single LOS band: {src_path}')
        source = src.read(1, masked=True).astype('float32').filled(np.nan)
        source[~np.isfinite(source)] = np.nan
        destination = np.full((height, width), NODATA, dtype=np.float32)
        reproject(
            source=source, destination=destination,
            src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
            dst_transform=transform, dst_crs=target_crs, dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )
        with rasterio.open(
            dst_path, 'w', driver='GTiff', dtype='float32', count=1,
            width=width, height=height, crs=target_crs, transform=transform,
            nodata=NODATA, compress='deflate',
        ) as dst:
            dst.write(destination, 1)
            dst.update_tags(**src.tags())
            dst.update_tags(1, **src.tags(1))
            dst.scales, dst.offsets = src.scales, src.offsets
            if src.descriptions[0]:
                dst.set_band_description(1, src.descriptions[0])
            if src.units[0]:
                dst.set_band_unit(1, src.units[0])
    print(f'Output: {dst_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-epsg', required=True, help='EPSG number or EPSG:code, e.g. 32735')
    parser.add_argument('--reference-raster', type=Path, required=True)
    parser.add_argument('--pattern', default='los_*.tif', help='Input filename pattern (default: los_*.tif)')
    args = parser.parse_args(argv)
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error('Input and output directories must differ.')
    target_crs = CRS.from_epsg(int(args.target_epsg.upper().removeprefix('EPSG:')))
    grid = target_grid(args.reference_raster, target_crs)
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f'No files matching {args.pattern!r} in {args.input_dir}')
    if any((args.output_dir / src.name).resolve() == args.reference_raster.resolve() for src in files):
        parser.error('An output would overwrite the reference raster.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Found {len(files)} rasters; target {target_crs}, size {grid[1]} x {grid[2]}. Units unchanged.')
    for src_path in files:
        reproject_raster(src_path, args.output_dir / src_path.name, target_crs, grid)


if __name__ == '__main__':
    main()
