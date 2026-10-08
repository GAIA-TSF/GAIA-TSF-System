"""Print Sentinel-2 extents and group images with identical bounds and CRS."""

import argparse
from collections import defaultdict
from pathlib import Path

import rasterio
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, help='Override Sentinel-2 directory.')
    parser.add_argument('--pattern', help='Override configured filename pattern.')
    parser.add_argument(
        '--config', type=Path,
        default=Path(__file__).resolve().parents[1] / 'config.yaml',
    )
    args = parser.parse_args()
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    inputs = config['amd']['inputs']['sentinel2']
    directory = args.directory or Path(config['project_dir']) / inputs['directory']
    paths = sorted(directory.glob(args.pattern or inputs.get('filename_pattern', '*.jp2')))
    if not paths:
        parser.error(f'No images found in {directory}')
    groups = defaultdict(list)
    print('Extent order: left, bottom, right, top (in raster CRS units)')
    for path in paths:
        with rasterio.open(path) as source:
            bounds = tuple(source.bounds)
            crs = str(source.crs)
        groups[(crs, bounds)].append(path.name)
        print(f'{path.name}: {bounds}, CRS={crs}')
    print(f'\n{len(paths)} images; {len(groups)} distinct exact extent/CRS groups.')
    for number, ((crs, bounds), names) in enumerate(groups.items(), 1):
        print(f'\nGroup {number}: {bounds}, CRS={crs}; {len(names)} images')
        for name in names:
            print(f'  {name}')


if __name__ == '__main__':
    main()
