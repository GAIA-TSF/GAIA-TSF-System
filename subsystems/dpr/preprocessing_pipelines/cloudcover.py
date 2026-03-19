from .base import BasePipeline
from osgeo import gdal
import os
import json
import numpy as np


class Sentinel2CloudCoverPipeline(BasePipeline):
    metadata = {
        'title': 'Sentinel-2 Cloud Cover',
        'abstract': 'Retrieve Sentinel-2 cloud cover and other quality parameters from the SCL band.'
                    'Input: JSON metadata file containing the path to a Sentinel-2 raster.'
                    'Output: Add SCL classes percentages and overall cloud cover percentage to the JSON metadata file.',
    }

    def __init__(self, metadata_path: str, path_key: str, scl_band: int = 13):
        """
        Initialize the pipeline with the path to a JSON metadata file.
        :param metadata_path: Path to the JSON file containing the 'path' key.
        :param path_key: Name of the dictionary key containing the path to the raster.
        :param scl_band: The index of the Sentinel-2 SCL band (default is band 13).
        """
        self.metadata_path = metadata_path
        self.path_key = path_key
        self.scl_band = scl_band
        self.metadata = self._load_json()
        self.raster_path = self._get_path(self.metadata, self.path_key)

        if not self.raster_path:
            raise KeyError("The 'path' key could not be found.")

        if not os.path.exists(self.raster_path):
            raise FileNotFoundError(f"Raster file not found at: {self.raster_path}")

    def _load_json(self):
        """Loads the JSON metadata file."""
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def _save_json(self):
        """Saves the current state of metadata back to the JSON file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

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

    def _process_slc_statistics(self):
        """
        Extracts the slc band, calculates pixel counts for each class,
        and appends the results to the JSON file.
        """
        # Register all GDAL drivers
        gdal.AllRegister()
        dataset = gdal.Open(self.raster_path, gdal.GA_ReadOnly)

        if dataset is None:
            raise Exception(f"Unable to open {self.raster_path}. Ensure it is a valid raster.")

        if self.scl_band > dataset.RasterCount or self.scl_band < 1:
            raise ValueError(f"Invalid band index {self.scl_band}. File has {dataset.RasterCount} bands.")

        # get slc band as array
        band = dataset.GetRasterBand(self.scl_band)
        band_data = band.ReadAsArray()

        if np.nanmax(band_data) > 12 or np.nanmax(band_data) < 0:
            raise ValueError(f"Invalid SCL values {np.nanmax(band_data)}. SCL values should be comprised between 1"
                             f"and 12.")

        # Get the unique values and their counts
        unique, counts = np.unique(band_data, return_counts=True)
        npix = np.sum(counts)

        # List of SLC band labels corresponding to pixel values 0 to 11
        slc_labels = ['SCL_no_data', 'SCL_saturated_or_defective', 'SCL_dark_areas',
                      'SCL_cloud_shadows', 'SCL_vegetation', 'SCL_non_vegetated', 'SCL_water',
                      'SCL_unclassified', 'SCL_cloud_medium_probability', 'SCL_cloud_high_probability',
                      'SCL_thin_cirrus', 'SCL_snow_or_ice']

        # Initialize a dictionary with the labels and 0s
        stats_dict = dict.fromkeys(slc_labels, 0)

        # get pixel percentages (float 0.0 - 1.0) for classes appearing in the SCL band
        for k, v in zip(unique, counts):
            stats_dict[slc_labels[k]] = round(v / npix, 4)

        # Sum of cloud classes (key named 'cloud_cover_pct' as in QCL subsystem)
        stats_dict['cloud_cover_pct'] = stats_dict['SCL_cloud_medium_probability'] + stats_dict[
            'SCL_cloud_high_probability'] + stats_dict['SCL_thin_cirrus']

        # add a no data key named 'null_pixel_pct' as in QCL subsystem
        stats_dict['null_pixel_pct'] = stats_dict['SCL_no_data']

        # Update metadata object
        self.metadata.update(stats_dict)

        # Save the updated metadata back to the file
        self._save_json()

    def run(self):
        self._process_slc_statistics()
