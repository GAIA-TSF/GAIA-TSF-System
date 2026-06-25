import os
import json
from pathlib import PosixPath

import numpy as np
from osgeo import gdal

from .base import PreprocessingBasePipeline


class Sentinel2CloudCoverPipeline(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-2 Cloud Cover',
        'abstract': 'Read metadata JSON file, retrieve Sentinel-2 scene path, compute cloud cover and other land cover'
        'classes from the SCL band, write the percentages to the metadata file.',
        'params': {
            'metadata_path': {
                'dtype': PosixPath,
                'description': 'Path to the JSON file containing the path key.',
            },
            'path_key': {
                'dtype': str,
                'description': 'Name of the metadata dictionary key containing the path to the raster.',
            },
            'scl_band': {
                'dtype': int,
                'description': 'The index of the Sentinel-2 SCL band (default is band 13).',
                'default': 13,
            },
        },
    }

    def _configure(self):
        """Initialize the processor with None values."""
        self._file_metadata = None

    def _load_json(self):
        """Loads the JSON metadata file."""
        with open(self._config['metadata_path'], 'r') as f:
            return json.load(f)

    def _save_json(self):
        """Saves the current state of metadata back to the JSON file."""
        with open(self._config['metadata_path'], 'w') as f:
            json.dump(self._file_metadata, f, indent=4)

    def _get_path(self, metadata, key):
        """
        Recursively searches the metadata dictionary for the path key and returns its value.
        """
        if isinstance(metadata, dict):
            if key in metadata:
                return metadata[key]
            for k, v in metadata.items():
                result = self._get_path(v, key)
                if result is not None:
                    return result
        return None

    def _process_scl_statistics(self, raster_path):
        """
        Extracts the SCL band, calculates pixel counts for each class,
        and appends the results to the JSON file.

        :param str raster_path: path to raster file to be processed
        """
        # Register all GDAL drivers
        gdal.AllRegister()
        dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)

        if dataset is None:
            raise Exception(
                f'Unable to open {raster_path}. Ensure it is a valid raster.'
            )

        if (
            self._config['scl_band'] > dataset.RasterCount
            or self._config['scl_band'] < 1
        ):
            raise ValueError(
                f'Invalid band index {self._config["scl_band"]}. File has {dataset.RasterCount} bands.'
            )

        # get slc band as array
        band = dataset.GetRasterBand(self._config['scl_band'])
        band_data = band.ReadAsArray()

        if np.nanmax(band_data) > 12 or np.nanmax(band_data) < 0:
            raise ValueError(
                f'Invalid SCL values {np.nanmax(band_data)}. SCL values should be comprised between 1'
                f'and 12.'
            )

        # Get the unique values and their counts
        unique, counts = np.unique(band_data, return_counts=True)
        npix = np.sum(counts)

        # List of SLC band labels corresponding to pixel values 0 to 11
        slc_labels = [
            'SCL_no_data',
            'SCL_saturated_or_defective',
            'SCL_dark_areas',
            'SCL_cloud_shadows',
            'SCL_vegetation',
            'SCL_non_vegetated',
            'SCL_water',
            'SCL_unclassified',
            'SCL_cloud_medium_probability',
            'SCL_cloud_high_probability',
            'SCL_thin_cirrus',
            'SCL_snow_or_ice',
        ]

        # Initialize a dictionary with the labels and 0s
        stats_dict = dict.fromkeys(slc_labels, 0)

        # get pixel percentages (float 0.0 - 1.0) for classes appearing in the SCL band
        for k, v in zip(unique, counts):
            stats_dict[slc_labels[k]] = round(v / npix, 4)

        # Sum of cloud classes (key named 'cloud_cover_pct' as in QCL subsystem)
        sum_cloud_classes = (
            stats_dict['SCL_cloud_medium_probability']
            + stats_dict['SCL_cloud_high_probability']
            + stats_dict['SCL_thin_cirrus']
        )
        cloud_cover_pct = {'cloud_cover_pct': sum_cloud_classes}

        # add a no data key named 'null_pixel_pct' as in QCL subsystem
        null_pixel_pct = {'null_pixel_pct': stats_dict['SCL_no_data']}

        # Update file metadata object
        self._file_metadata['SCL_classes_pct'] = stats_dict
        self._file_metadata.update(cloud_cover_pct)
        self._file_metadata.update(null_pixel_pct)

        # Save the updated metadata back to the file
        self._save_json()

    def _run(self):
        """ """
        if not os.path.exists(self._config['metadata_path']):
            raise FileNotFoundError(
                f'Metadata file not found at: {self._config["metadata_path"]}'
            )

        if self._config['path_key'] is None:
            raise KeyError("The 'path' key could not be found.")

        if self._config['scl_band'] < 0:
            raise ValueError(
                f'Invalid SCL band index value {self._config["scl_band"]}. SCL band index should be greater than 0.'
            )

        if not isinstance(self._config['scl_band'], int):
            # Raise a ValueError with a descriptive message
            raise ValueError(
                f'Invalid SCL band index value {self._config["scl_band"]}. Input value must be an integer.'
            )

        self._file_metadata = self._load_json()
        raster_path = self._get_path(self._file_metadata, self._config['path_key'])
        # locate raster associated to metadata
        if isinstance(raster_path, str):
            pass
        elif isinstance(raster_path, list):
            if len(raster_path) > 1:
                raise ValueError(
                    f'Error parsing metadata. The input must be a single multi_band raster. The metadata contains'
                    f'paths for {len(raster_path)} separate files.'
                )
            else:
                raster_path = raster_path[0]
        else:
            raise ValueError(
                'Error retrieving raster path from metadata. The "source_paths" key is neither a string '
                'or a list.'
            )

        if not os.path.exists(raster_path):
            raise FileNotFoundError(f'Raster file not found at: {raster_path}')

        self._process_scl_statistics(raster_path)
