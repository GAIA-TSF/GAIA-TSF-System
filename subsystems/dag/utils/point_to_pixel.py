

import argparse

import rasterio
from rasterio.transform import rowcol
import geopandas as gpd

"""
Usage: 
python3 subsystems/dag/utils/point_to_pixel.py \
    --raster /path/to/raster.tif \
    --vector /path/to/points.gpkg 
"""


def load_raster(raster_path):
    print(f'[INFO] Loading raster: {raster_path}')
    return rasterio.open(raster_path)


def load_points(vector_path):
    print(f'[INFO] Loading vector: {vector_path}')
    gdf = gpd.read_file(vector_path)

    if gdf.empty:
        raise ValueError('Vector file contains no features')

    if not all(gdf.geometry.type == 'Point'):
        raise ValueError('All geometries must be POINT')

    return gdf


def reproject_points(gdf, target_crs):
    if gdf.crs != target_crs:
        print('[INFO] Reprojecting points to raster CRS')
        gdf = gdf.to_crs(target_crs)

    return gdf


def get_pixel_indices(dataset, gdf):
    print("Get pixel indices for points") 
    results = []

    for idx, row in gdf.iterrows():
        x = row.geometry.x
        y = row.geometry.y

        try:
            row_idx, col_idx = rowcol(dataset.transform, x, y)

            # Check bounds
            if (
                0 <= row_idx < dataset.height
                and 0 <= col_idx < dataset.width
            ):
                print(f'[OK] Point {idx}: (x={x}, y={y}) → pixel_y={row_idx}, pixel_x={col_idx}')

                results.append(
                    {
                        'id': idx,
                        'x': x,
                        'y': y,
                        'pixel_y': row_idx,
                        'pixel_x': col_idx,
                        'in_bounds': True,
                    }
                )
            else:
                print(f'[WARN] Point {idx} outside raster extent')

                results.append(
                    {
                        'id': idx,
                        'x': x,
                        'y': y,
                        'pixel_y': None,
                        'pixel_x': None,
                        'in_bounds': False,
                    }
                )

        except Exception as e:
            print(f'[ERROR] Failed for point {idx}: {e}')

    return results


def main():
    parser = argparse.ArgumentParser(description='Convert point(s) to pixel coordinates')

    parser.add_argument('--raster', required=True, help='Path to raster (.tif)')
    parser.add_argument('--vector', required=True, help='Path to vector (.gpkg)')

    args = parser.parse_args()

    dataset = load_raster(args.raster)
    gdf = load_points(args.vector)

    # gdf = reproject_points(gdf, dataset.crs)

    results = get_pixel_indices(dataset, gdf)

    print('\n[RESULTS]')
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
