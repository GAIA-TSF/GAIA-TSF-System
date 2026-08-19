import os
import json
import zipfile
from pathlib import Path, PosixPath

from .base import PreprocessingBasePipeline


class Sentinel2ZipStacAssets(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-2 Zip STAC Assets',
        'abstract': 'Create a zip archive containing a STAC metadata file and its assets.',
        'params': {
            'input_json': {
                'dtype': PosixPath,
                'description': 'Path to the json STAC metadata file.',
            },
            'output_folder': {
                'dtype': PosixPath,
                'description': 'Path to the folder where the zipped items will be stored.',
            },
            'delete_original': {
                'dtype': bool,
                'description': 'If true delete input STAC metadata file and its assets after the zip archive is '
                'created.',
                'default': False,
            },
        },
    }

    def _initialize(self):
        """
        Initialize the processor.
        """
        self.input_json = self._config['input_json']
        self.output_folder = self._config['output_folder']
        self.delete_original = self._config['delete_original']
        self.metadata = None
        self.asset_paths = None
        self.asset_count = None
        self.zip_filename = None
        self.zip_path = None

    def _list_stac_assets(self):
        """
        Parses a STAC metadata item for raster assets and return their filenames and count.
        """
        # Check that the item is not a collection
        # Normalize data structure into a list of items
        stac_type = self.metadata.get('type')
        if stac_type == 'Feature':  # single STAC Item
            items = [self.metadata]
        elif 'assets' in self.metadata:  # Collection with assets directly attached
            items = [self.metadata]
        else:
            self.logger.error(
                f'Provided JSON does not directly contain STAC assets: {self.metadata}'
            )

        # Common raster identifiers based on STAC specifications
        raster_types = {
            'image/tiff',
            'image/vnd.stac.geotiff',
            'image/jp2',
            'image/jpeg',
            'image/png',
        }
        raster_roles = {'data', 'visual', 'overview', 'reflectance', 'classification'}

        self.asset_count = 0
        self.asset_paths = []
        # Iterate through items and extract assets
        for item in items:
            assets = item.get('assets', {})
            if not assets:
                continue

            for asset_key, asset_info in assets.items():
                href = asset_info.get('href')
                if not href:
                    continue

                media_type = asset_info.get('type', '')
                roles = asset_info.get('roles', [])

                # Determine if the asset is a raster layer
                is_raster = (media_type in raster_types) or any(
                    role in raster_roles for role in roles
                )

                # Extract the filename
                if is_raster:
                    clean_url = href.split('?')[0]
                    filename = os.path.basename(clean_url)
                    self.asset_paths.append(filename)
                    self.asset_count += 1

    def _zip_stac_assets(self):
        """
        Generates an archive from the input paths of the json metadata file and assets.
        """
        with zipfile.ZipFile(
            self.zip_path, mode='w', compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            for raw_path in self.asset_paths:
                path = raw_path.resolve()
                if path.exists():
                    # Add single file to root of ZIP archive
                    zipf.write(path, arcname=path.name)
                    self.logger.info(f'- Added file to zip: {path.name}')
                else:
                    self.logger.warning(f'- Skipping file (not found): {path.name}')

    def _run(self):
        self._initialize()
        # Preliminary checks
        # Ensure input paths are Path objects
        if isinstance(self.input_json, Path) is False:
            self.input_json = Path(self.input_json).resolve()
        if isinstance(self.output_folder, Path) is False:
            self.output_folder = Path(self.output_folder).resolve()
        # Check if input json file exists
        if not self.input_json.is_file():
            self.logger.error(f'File {self.input_json} does not exist.')
        # Ensure output folder exists
        self._config['output_folder'].mkdir(parents=True, exist_ok=True)

        # open json file
        with open(self.input_json, 'r') as f:
            self.metadata = json.load(f)

        # locate raster associated to metadata, return assets count and paths
        self.logger.info(f'Parsing metadata from {self.input_json.stem}')
        self._list_stac_assets()
        # Check that the list of assets is not empty and proceed with creating a zip archive
        if self.asset_count == 0:
            self.logger.warning(
                'The paths to the STAC assets could not be retrieved. No zip archive will be created.'
            )
        else:
            # set path to output zip
            self.zip_filename = f'{self.input_json.stem}.zip'
            self.zip_path = self.output_folder / self.zip_filename
            self.logger.info(
                f'Packaging metadata & assets into zip folder: {self.zip_path}'
            )
            # Add JSON path to asset paths and create the archive
            self.asset_paths = [self.input_json] + [
                Path(self.input_json.parent / p) for p in self.asset_paths
            ]
            # also add xml files if assets are OpenJPEG
            for path in self.asset_paths:
                if path.suffix.lower() == '.jp2' or path.suffix.lower() == '.jpeg':
                    xml_path = path.with_suffix('.jp2.aux.xml')
                    if os.path.exists(xml_path):
                        self.asset_paths.append(xml_path)
            self._zip_stac_assets()
            # Check that the archive was created and only delete the originals if true
            if os.path.exists(self.zip_path):
                with zipfile.ZipFile(self.zip_path, 'r') as zf:
                    self.logger.info('Successfully packaged zip archive.')
                    for zip_info in zf.infolist():
                        self.logger.info(
                            f' - {zip_info.filename} ({zip_info.file_size} bytes)'
                        )
                if self.delete_original is True:
                    for raw_path in self.asset_paths:
                        path = raw_path.resolve()
                        if path.exists():
                            self.logger.info(f'Deleting original file: {path}')
                            path.unlink()
            else:
                self.logger.warning('The zip archive has not been created.')
