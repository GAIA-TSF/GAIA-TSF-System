import os
import json
import glob
from pathlib import PosixPath

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, osr, ogr
from scipy.ndimage import label
from shapely.geometry import box as sbox

from .base import DataAnalysisBasePipeline


class Sentinel2WaterMaskingPipeline(DataAnalysisBasePipeline):
    metadata = {
        'title': 'Sentinel-2 Water Masking',
        'abstract': 'This pipeline generates water masks for Sentinel-2 scenes.'
                    'Main processing steps include:'
                    '(1) Parsing metadata from input folder and create a list of scenes. The user can apply a temporal'
                    '    filter (optional) using "start_date" and "end_date" parameters.'
                    '    (A) Optional. The user can provide a vector file (e.g., *.shp, *.gpkg) with digitized water'
                    '        bodies (it is recommended to map the maximum extent of the water features) using the'
                    '        "input_water_mask" parameter. This shapefile will be converted into a labeled array.'
                    '    (B) Default. Filtered scenes (according to the percentage of cloud/snow/dark pixels using'
                    '        "max_cloud_snow_dark" parameter) will be aggregated into a median raster. A water mask'
                    '        will be derived by thresholding spectral indices. This mask is then converted into a'
                    '        labeled array. This method is sensitive to topographic shadows. Winter months should be'
                    '        avoided. The optional "input_months" parameter can be used to filter scenes by months.'
                    '(2) Generation of a water mask for each Sentinel-2 scenes. Thresholding on spectral indices'
                    '    (similar to step 2B) will be used to check the global water mask validity for a given scene.'
                    '    Pixels that do not fulfill thresholding conditions are removed to create a scene mask.'
                    '    If the scene mask contains valid pixels, it will be appended to the Sentinel-2 bands and the'
                    '    merged raster will be exported to the output folder.',
        'params': {
            'input_folder': {
                'dtype': PosixPath,
                'description': 'Path to the folder containing the processed (clipped) Sentinel-2 SAFE products',
            },
            'output_folder': {
                'dtype': PosixPath,
                'description': 'Directory where the processed data will be saved.',
            },
            'output_format': {
                'dtype': str,
                'description': 'Format of the output rasters: "tiff" or "jp2".',
                'default': 'jp2',
            },
            'max_cloud_snow_dark': {
                'dtype': float,
                'description': 'Parameter used to filter scenes based on the maximum percentage of pixels containing '
                               'clouds/snow/shadows. Ratio between 0 and 1. Scenes with a ratio above this value will '
                               'be filtered out',
                'default': 0.2,
            },
            'threshold_parameters': {
                'dtype': dict,
                'description': 'Thresholds used to tag water bodies:'
                               'Shadow Index is computed for NIR bands and range between 0 (light pixels) and'
                               '    1 (darkest pixel). Water bodies absorb light in NIR and are thus expected to'
                               '    have shadow index values close to 1.'
                               'Spectral Angle is used to filter out pixels based on provided reference spectra (see'
                               '    "reference_spectra" parameter). Smaller angles means greater spectral similarity.'
                               '    Value in radians.'
                               'VNIR regression slope and intercept are parameters based on the linear regression of'
                               '    bands 1, 2, 6, 7, 8 and 8A. Water bodies have a relatively flat (or slightly'
                               '    increasing/decreasing) profile in these bands, with slopes in the range 0-1 and'
                               '    intercept close to 0. On the other hand soils and vegetation profiles have higher'
                               '    slopes and strongly negative intercepts.'
                               'Band 2 reflectance threshold (in scale 0-10000) is mainly used to filter out clouds.'
                               'SWIR reflectance threshold (in scale 0-10000) is used to filter water bodies based on'
                               '    the strong water absorption in these wavelengths.',
                'default': {
                    'shadow index': 0.9,
                    'spectral angle': 0.15,
                    'vnir regression slope': 1,
                    'vnir regression intercept': -500,
                    'band 2': 2000,
                    'swir reflectance': 1000,
                },
            },
            'input_water_mask': {
                'dtype': PosixPath,
                'description': 'The user can provide a vector file (e.g., .shp, .gpkg) for water bodies '
                               'instead of relying on the pipeline filtering.',
                'required': False,
            },
            'input_months': {
                'dtype': list,
                'description': 'List of months (e.g., [1, 2, 5, 6]) to use for water masking.',
                'required': False,
            },
            'start_date': {
                'dtype': str,
                'description': 'Starting date for temporal filtering as "YYYY-MM-DD".',
                'required': False,
            },
            'end_date': {
                'dtype': str,
                'description': 'End date for temporal filtering as "YYYY-MM-DD".',
                'required': False,
            },
            'minimum_water_area_pixels': {
                'dtype': int,
                'description': 'Minimum size of extracted water bodies in pixels.',
                'default': 4,
            },
            'reference_spectra': {
                'dtype': list,
                'description': 'List of reference spectra used as input for discarding pixels based on spectral'
                               'similarity (i.e. Spectral Angle Mapper algorithm). Spectral angle threshold is set in'
                               'the "threshold_parameters" dictionary and pixels below this threshold will be not be'
                               'tagged as water pixels. NOTE: Sentinel-2 Band 9 is not used and should not be provided!'
                               '(i.e. refernce spectra should contain 11 bands).',
                'default': [
                    [
                        261.0,
                        498.5,
                        748.5,
                        971.0,
                        1385.0,
                        2026.0,
                        2310.0,
                        2550.0,
                        2691.5,
                        2933.0,
                        1845.0,
                    ],
                    [
                        32.0,
                        124.0,
                        285.0,
                        183.0,
                        531.0,
                        1603.0,
                        1903.5,
                        2018.0,
                        2081.0,
                        921.0,
                        432.0,
                    ],
                ],
            },
        },
    }

    def _configure(self):
        """Initialize the processor with None values."""
        self.global_watermask = None
        self.scenes_metadata = None
        self.scenes_metadata_filtered = None

    def _parse_metadata(self):
        """
        Retrieve metadata from json files and store them to a dictionary ('scenes_metadata'). Also perform a filtering
        ('scenes_metadata_filtered') based on the maximum percentage of clouds/snow/dark pixels (max_cloud_snow_dark
        parameter).
        """
        self.logger.info(
            f'Parsing metadata. Input folder: {self._config["input_folder"]}.'
        )

        json_files = glob.glob(
            os.path.join(self._config['input_folder'], '**/*.json'), recursive=True
        )
        if len(json_files) == 0:
            raise FileNotFoundError(f'Error parsing {self._config["input_folder"]}')

        self.logger.info(f'--- The input folder contains {len(json_files)} entries.')

        records = []
        for json_path in json_files:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # locate raster associated to metadata
            if isinstance(data.get('source_paths'), str):
                source_path = data.get('source_paths')
            elif isinstance(data.get('source_paths'), list):
                if len(data.get('source_paths')) > 1:
                    raise ValueError(
                        f'Error parsing metadata. The input must be a single multi_band raster. The metadata contains'
                        f'paths for {len(data.get("source_paths"))} separate files.')
                else:
                    source_path = data.get('source_paths')[0]
            else:
                raise ValueError('Error parsing raster path from metadata. The "source_paths" key is neither a string '
                                 'or a list.')
            if os.path.exists(source_path):
                raster_path = source_path
            elif os.path.exists(json_path.replace('.json', '.jp2')):
                raster_path = json_path.replace('.json', '.jp2')
            elif os.path.exists(json_path.replace('.json', '.tiff')):
                raster_path = json_path.replace('.json', '.tiff')
            else:
                raise FileNotFoundError(
                    f'Could not locate the raster associated with {json_path}'
                )

            scl_stats = data.get('SCL_classes_pct', {})
            record = {
                'DATATAKE_SENSING_START': data.get('DATATAKE_SENSING_START'),
                'source_path': raster_path,
                'metadata_path': json_path,
                'SCL_snow_or_ice': scl_stats.get('SCL_snow_or_ice', 0),
                'cloud_cover_pct': data.get('cloud_cover_pct', 0),
                'SCL_cloud_shadows': scl_stats.get('SCL_cloud_shadows', 0),
                'SCL_dark_areas': scl_stats.get('SCL_dark_areas', 0),
            }
            records.append(record)

        self.scenes_metadata = pd.DataFrame(records)

        if not self.scenes_metadata.empty:
            # Temporal filtering
            if (
                    self._config['start_date'] is not None
                    or self._config['end_date'] is not None
            ):
                # Ensure we are working with a copy to avoid SettingWithCopy warnings
                df = self.scenes_metadata.copy()

                # Convert column to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(
                        df['DATATAKE_SENSING_START']
                ):
                    df['DATATAKE_SENSING_START'] = pd.to_datetime(
                        df['DATATAKE_SENSING_START']
                    )

                # Apply start date filter
                if self._config['start_date'] is not None:
                    if pd.to_datetime(self._config['start_date']).tz is None:
                        start_dt = pd.to_datetime(
                            self._config['start_date']
                        ).tz_localize('UTC')
                    else:
                        start_dt = pd.to_datetime(self._config['start_date'])
                    df = df[df['DATATAKE_SENSING_START'] >= start_dt]
                    self.logger.info(f'--- Applied start date filter: {start_dt}')

                # Apply end date filter
                if self._config['end_date'] is not None:
                    if pd.to_datetime(self._config['end_date']).tz is None:
                        end_dt = pd.to_datetime(self._config['end_date']).tz_localize(
                            'UTC'
                        )
                    else:
                        end_dt = pd.to_datetime(self._config['end_date'])
                    df = df[df['DATATAKE_SENSING_START'] <= end_dt]
                    self.logger.info(f'--- Applied end date filter: {end_dt}')

                # Update the filtered metadata attribute
                self.scenes_metadata = df.reset_index(drop=True)
                self.logger.info(
                    f'--- Temporal filter applied. {len(self.scenes_metadata)} entries remains.'
                )

            # Cloud/Snow/Dark pixels filtering
            self.scenes_metadata['cloud_snow_dark'] = (
                    self.scenes_metadata['SCL_dark_areas']
                    + self.scenes_metadata['SCL_snow_or_ice']
                    + self.scenes_metadata['SCL_cloud_shadows']
                    + self.scenes_metadata['cloud_cover_pct']
            )

            self.scenes_metadata_filtered = self.scenes_metadata[
                self.scenes_metadata['cloud_snow_dark']
                < self._config['max_cloud_snow_dark']
                ].copy()

            self.scenes_metadata_filtered['DATATAKE_SENSING_START'] = pd.to_datetime(
                self.scenes_metadata_filtered['DATATAKE_SENSING_START']
            )
            self.scenes_metadata_filtered = self.scenes_metadata_filtered.sort_values(
                'DATATAKE_SENSING_START'
            ).reset_index(drop=True)

            self.logger.info(
                f'Cloud/Snow/Dark entries filtered out. {len(self.scenes_metadata_filtered)} entries '
                f'remains.'
            )

    def _get_geotransform(self):
        """
        Retrieve spatial data from first data file in the input folder. Assumes that all scenes in the input folder share
        the same EPSG and geotransform. Only used if a vector file is provided (input_water_mask is not None).
        """
        self.logger.info('Retrieving spatial data from input folder')

        raster_path = os.path.abspath(self.scenes_metadata_filtered['source_path'][0])
        if os.path.exists(raster_path) is False:
            raise FileNotFoundError(
                f'Retrieval of EPSG and geotransform failed: '
                f'{raster_path} either cannot be accessed or does not exists.'
            )
        ds = gdal.Open(raster_path)
        if ds:
            self.geotransform = ds.GetGeoTransform()
            self.projection = ds.GetProjection()
            self.shape = (ds.RasterYSize, ds.RasterXSize)

            # Get EPSG
            srs = osr.SpatialReference(wkt=self.projection)
            self.epsg = srs.GetAttrValue('AUTHORITY', 1)

            # Calculate Bounding Box
            gt = self.geotransform
            min_x = gt[0]
            max_y = gt[3]
            max_x = gt[0] + gt[1] * ds.RasterXSize
            min_y = gt[3] + gt[5] * ds.RasterYSize
            self.bounds = (min_x, min_y, max_x, max_y)
            ds = None
        else:
            raise OSError(
                f'Retrieval of EPSG and geotransform failed: could not read raster {raster_path}'
            )

    def _calculate_spectral_angle(self, reference_spectrum, dataset, bands=None):
        """
        Calculates the spectral angle between a reference spectrum and a stack of Sentinel-2 bands.
        :param reference_spectrum: List of reflectance values from a reference spectrum.
        :param dataset: Opened gdal dataset.
        :param bands: OPTIONAL. List of bands to be used (indices in gdal format, 1 for Band 1, etc.)
        """
        # Prepare data for SAM (uses all bands if no specific bands are provided)
        if bands is None:
            bands = list(range(1, dataset.RasterCount))

        sam_input = []
        for b in bands:
            band = dataset.GetRasterBand(b)
            sam_input.append(band.ReadAsArray().astype(np.float32))

        sam_stack = np.stack(sam_input, axis=-1)

        rows, cols, bands = sam_stack.shape
        spectra_flat = sam_stack.reshape((rows * cols, bands))

        dot_product = np.dot(spectra_flat, reference_spectrum)
        norm_product = np.linalg.norm(spectra_flat, axis=1) * np.linalg.norm(
            reference_spectrum
        )

        # Prevent division by zero and handle out-of-range for arccos
        cosine_sim = np.divide(
            dot_product,
            norm_product,
            out=np.zeros_like(dot_product),
            where=norm_product != 0,
        )
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

        sam_1d = np.arccos(cosine_sim)
        return sam_1d.reshape((rows, cols))

    def _calculate_vectorized_regression(self, dataset):
        """
        Calculates regression parameters (slope, intercept) for a gdal dataset.
        """
        target_bands = [1, 2, 6, 7, 8, 9]
        band_data = []
        for b in target_bands:
            band = dataset.GetRasterBand(b)
            if band:
                band_data.append(band.ReadAsArray().astype(np.float32))

        if not band_data:
            return None, None

        stack = np.array(band_data)
        n_bands, n_rows, n_cols = stack.shape
        y = stack.reshape(n_bands, -1)
        # x = np.array(target_bands, dtype=np.float32)
        x = [443, 490, 740, 783, 842, 865]  # S2 wavelengths for target bands

        x_mean = np.mean(x)
        y_mean = np.mean(y, axis=0)
        x_diff = x - x_mean

        numerator = np.dot(x_diff, y - y_mean)
        denominator = np.sum(x_diff ** 2)

        m = numerator / denominator
        b = y_mean - (m * x_mean)

        slope_map = m.reshape((n_rows, n_cols))
        intercept_map = b.reshape((n_rows, n_cols))

        return np.nan_to_num(slope_map, nan=-9999), np.nan_to_num(
            intercept_map, nan=-9999
        )

    def _calculate_shadow_index(self, dataset, indices):
        """
        Calculates the shadow index (NIR bands) for a gdal dataset.
        :param dataset: Opened gdal dataset.
        :param indices: List containing the indices of bands to be used (in gdal format, 1 for Band 1, etc.).
                        Minimum 2 values.
        """
        bands = []
        for i in indices:
            b = dataset.GetRasterBand(i).ReadAsArray().astype(np.float32)
            # Simple normalization (assuming 10000 reflectance scale)
            bands.append(b / 10000.0)

        term = 1.0 - bands[0]
        for i in range(1, len(bands)):
            term = term * (1.0 - bands[i])

        term = np.maximum(term, 0)
        si = (term ** (1.0 / len(bands))).astype('float32')
        return si

    def _get_swir_reflectance(self, dataset):
        """
        Calculates the average swir reflectance for a gdal dataset.
        """
        swir_reflect = (
                               dataset.GetRasterBand(11).ReadAsArray()
                               + dataset.GetRasterBand(12).ReadAsArray()
                       ) / 2
        return swir_reflect

    def _label_features(self, binary_mask):
        # Label components (we want the largest body to be 1)
        # Using scipy.ndimage.label
        labeled_array, num_features = label(binary_mask)

        if self._config['minimum_water_area_pixels']:
            # Get unique labels and their corresponding pixel counts
            labels, counts = np.unique(labeled_array, return_counts=True)
            # Filter out the background label (0)
            non_bg_indices = labels > 0
            feature_labels = labels[non_bg_indices]
            feature_counts = counts[non_bg_indices]
            # Get labels that do not meet minimum area threshold
            small_labels = feature_labels[
                feature_counts < self._config['minimum_water_area_pixels']
                ]
            # Create a copy of original input array
            filtered_mask = labeled_array.copy()
            # Mask out small features by setting their pixels to 0 (background) and re-label the array.
            if small_labels.size > 0:
                filtered_mask[np.isin(filtered_mask, small_labels)] = 0
                binary_mask = (filtered_mask > 0).astype('uint8')
                labeled_array, num_features = label(binary_mask)

        # Sort labels by size to maintain consistency with the 'rank' logic
        if num_features > 1:
            sizes = np.bincount(labeled_array.ravel())
            # sizes[0] is background, ignore it
            sorted_labels = np.argsort(sizes[1:])[::-1] + 1

            # Create a ranked mask (1 = largest water body)
            ranked_mask = np.zeros_like(labeled_array, dtype=np.uint16)
            for rank, label_id in enumerate(sorted_labels, start=1):
                ranked_mask[labeled_array == label_id] = rank
        else:
            ranked_mask = labeled_array.copy()

        return ranked_mask

    def _process_vector_watermask(self):
        """
        Ingests vector, reprojects, crops to bounds, and creates a size-labeled binary mask.
        """

        self.logger.info('Generating global water mask from vector file.')

        if not self.bounds:
            self.logger.info('Bounding box is missing.')

        # Load vector
        gdf = gpd.read_file(self._config['input_water_mask'])

        # Reproject if necessary
        target_crs = f'EPSG:{self.epsg}'
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Crop to Bounding Box
        bbox_geom = sbox(*self.bounds)
        gdf = gdf.clip(bbox_geom)

        if gdf.empty:
            self.logger.info('Could not parse the input water mask vectors.')

        # Create an in-memory layer for GDAL to rasterize
        driver = gdal.GetDriverByName('MEM')
        target_ds = driver.Create('', self.shape[1], self.shape[0], 1, gdal.GDT_UInt16)
        target_ds.SetGeoTransform(self.geotransform)
        target_ds.SetProjection(self.projection)

        # Create an OGR layer from the GeoDataFrame
        mem_driver = ogr.GetDriverByName('Memory')
        mem_ds = mem_driver.CreateDataSource('mem_ds')
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.projection)
        layer = mem_ds.CreateLayer('lakes', srs, ogr.wkbPolygon)

        # Populate the memory layer with features
        for i, row in gdf.iterrows():
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(ogr.CreateGeometryFromWkb(row.geometry.wkb))
            layer.CreateFeature(feature)
            feature = None

        # Rasterize
        gdal.RasterizeLayer(target_ds, [1], layer)

        binary_mask = (target_ds.GetRasterBand(1).ReadAsArray() > 0).astype('uint8')

        self.global_watermask = self._label_features(binary_mask)

        # Cleanup
        target_ds = None
        mem_ds = None

        self.logger.info('Global water mask generated successfully.')

    def _process_watermask(self):
        """
        Creates a median water mask by processing scenes listed in scenes_metadata_filtered.
        """

        self.logger.info('Generating global water mask from median raster.')

        if self.scenes_metadata_filtered is None or self.scenes_metadata_filtered.empty:
            self.logger.info(
                'No filtered scenes available to compute median water mask.'
            )
            return

        df = self.scenes_metadata_filtered.copy()

        # Filter by months if requested
        if self._config['input_months']:
            # Ensure DATATAKE_SENSING_START is datetime (already handled in _parse_metadata, but being safe)
            df['month'] = df['DATATAKE_SENSING_START'].dt.month
            df = df[df['month'].isin(self._config['input_months'])]
            self.logger.info(
                f'Filtered {len(df)} scenes matching months: {self._config["input_months"]}'
            )

            if df.empty:
                self.logger.error('No scenes found for the selected month range.')

        self.logger.info(f'Computing median raster from {len(df)} scenes...')

        # Initialize stack for median calculation
        stack = []

        # Stacking rasters.
        # Note: Depending on memory, we might need to handle this differently (process by bands?) for very large stacks.
        for _, row in df.iterrows():
            path = os.path.abspath(row['source_path'])
            ds = gdal.Open(path)
            if ds is None:
                continue
            stack.append(ds.ReadAsArray().astype(np.int16))
            ds = None
        if not stack:
            self.logger.error(
                'Failed to read data from any scenes for median raster generation.'
            )

        # Calculate pixel-wise median
        stack = np.array(stack)
        median = np.median(stack, axis=0)

        # We use the first scene to get raster dimensions
        first_data_file = os.path.abspath(df.iloc[0]['source_path'])
        ds_ref = gdal.Open(first_data_file)
        rows, cols = ds_ref.RasterYSize, ds_ref.RasterXSize

        # Save median as a temporary GDAL dataset
        # We use the 'MEM' driver to keep the dataset in RAM
        mem_driver = gdal.GetDriverByName('MEM')
        median_ds = mem_driver.Create('median', cols, rows, 12, gdal.GDT_Float32)
        median_ds.SetGeoTransform(self.geotransform)
        median_ds.SetProjection(self.projection)

        # Copy original bands
        for i in range(1, 12 + 1):
            median_ds.GetRasterBand(i).WriteArray(median[i - 1])

        si = self._calculate_shadow_index(median_ds, [6, 7, 8, 9])
        slope, intercept = self._calculate_vectorized_regression(median_ds)
        swir_reflect = self._get_swir_reflectance(median_ds)
        band2 = median_ds.GetRasterBand(2).ReadAsArray().astype(np.int16)

        sam_bands_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
        angles = []
        for i in range(len(self._config['reference_spectra'])):
            sam = self._calculate_spectral_angle(
                self._config['reference_spectra'][i], median_ds, sam_bands_idx
            )
            angles.append(sam)
        min_angles = np.nanmin(np.array(angles), axis=0)

        # Spectral refinement condition (pixels to be REMOVED from the global water mask)
        binary_mask = (
                (si > self._config['threshold_parameters']['shadow index'])
                & (slope < self._config['threshold_parameters']['vnir regression slope'])
                & (
                        intercept
                        > self._config['threshold_parameters']['vnir regression intercept']
                )
                & (swir_reflect < self._config['threshold_parameters']['swir reflectance'])
                & (min_angles > self._config['threshold_parameters']['spectral angle'])
                & (band2 < self._config['threshold_parameters']['band 2'])
        ).astype(np.uint8)

        self.global_watermask = self._label_features(binary_mask)

        self.logger.info('Global water mask generated successfully.')

    def _process_scenes(self):
        """
        Iterates through data files, update the global water mask based on spectral indices, appends mask to data files
        and exports merged results to output folder.
        """
        self.logger.info('Generating water masks for filtered scenes.')
        self.logger.info(
            f'Merged raster will be exported to {self._config["output_folder"]}'
        )

        if (
                self._config['start_date'] is not None
                or self._config['end_date'] is not None
        ):
            self.logger.info(f'Temporal filter start = {self._config["start_date"]}')
            self.logger.info(f'Temporal filter end = {self._config["end_date"]}')

        if self._config['input_months']:
            self.logger.info(f'Month filter = {self._config["input_months"]}')
            self.scenes_metadata_filtered['month'] = self.scenes_metadata_filtered[
                'DATATAKE_SENSING_START'
            ].dt.month
            indices = self.scenes_metadata_filtered['month'].isin(
                self._config['input_months']
            )
            self.scenes_metadata_filtered = self.scenes_metadata_filtered[indices]

        data_files = list(self.scenes_metadata_filtered['source_path'])

        for data_path in data_files:
            self.logger.info(f'--- Processing {os.path.basename(data_path)}')
            data_path = os.path.abspath(data_path)
            gdal_dataset = gdal.Open(data_path)
            if gdal_dataset is None:
                return None

            si = self._calculate_shadow_index(gdal_dataset, [6, 7, 8, 9])
            scl = gdal_dataset.GetRasterBand(13).ReadAsArray().astype(np.int16)
            slope, intercept = self._calculate_vectorized_regression(gdal_dataset)
            swir_reflect = self._get_swir_reflectance(gdal_dataset)
            band2 = gdal_dataset.GetRasterBand(2).ReadAsArray().astype(np.int16)

            sam_bands_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
            angles = []
            for i in range(len(self._config['reference_spectra'])):
                sam = self._calculate_spectral_angle(
                    self._config['reference_spectra'][i], gdal_dataset, sam_bands_idx
                )
                angles.append(sam)
            min_angles = np.nanmin(np.array(angles), axis=0)

            # Spectral refinement condition (pixels to be REMOVED from the global water mask)
            remove_conditions = (
                    (si < self._config['threshold_parameters']['shadow index'])
                    | (
                            slope
                            > self._config['threshold_parameters']['vnir regression slope']
                    )
                    | (
                            intercept
                            < self._config['threshold_parameters']['vnir regression intercept']
                    )
                    | (
                            swir_reflect
                            > self._config['threshold_parameters']['swir reflectance']
                    )
                    | (min_angles < self._config['threshold_parameters']['spectral angle'])
                    | (band2 > self._config['threshold_parameters']['band 2'])
                    | (scl >= 8)
                    | (scl <= 1)
            )

            refined_mask = self.global_watermask.copy()
            refined_mask[remove_conditions] = 0

            if np.nansum(refined_mask) > 0:
                # Export if mask contains valid pixels
                # Note: we create a Geotiff first and convert it to OpenJPEG if requested by config.
                out_name = os.path.basename(data_path)
                file_name, file_extension = os.path.splitext(out_name)
                out_name = out_name.replace(file_extension, '.tiff')
                creation_options = ['COMPRESS=DEFLATE', 'TILED=YES', 'PREDICTOR=2']
                out_path = os.path.join(self._config['output_folder'], out_name)

                driver = gdal.GetDriverByName('Gtiff')
                out_ds = driver.Create(
                    out_path,
                    gdal_dataset.RasterXSize,
                    gdal_dataset.RasterYSize,
                    gdal_dataset.RasterCount + 1,
                    gdal.GDT_UInt16,
                    options=creation_options,
                )
                out_ds.SetGeoTransform(gdal_dataset.GetGeoTransform())
                out_ds.SetProjection(gdal_dataset.GetProjection())

                # Copy original bands
                for i in range(1, gdal_dataset.RasterCount + 1):
                    out_ds.GetRasterBand(i).WriteArray(
                        gdal_dataset.GetRasterBand(i).ReadAsArray()
                    )

                # Add updated mask as last band
                out_ds.GetRasterBand(gdal_dataset.RasterCount + 1).WriteArray(
                    refined_mask
                )

                # clean up
                out_ds.FlushCache()
                out_ds = None

                if os.path.exists(out_path) is False:
                    raise FileNotFoundError(
                        f'Could not create output raster {out_name}'
                    )

                if self.driver_name == 'JP2OpenJPEG':
                    self._tiff_to_jp2(out_path)
                    file_name, file_extension = os.path.splitext(out_path)
                    out_path = out_path.replace(file_extension, '.jp2')

                # Copy and update metadata
                _, file_extension = os.path.splitext(data_path)
                json_path = data_path.replace(file_extension, '.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    rows, cols = gdal_dataset.RasterYSize, gdal_dataset.RasterXSize
                    metadata['water_mask_pct'] = round(
                        np.nansum((refined_mask > 0).astype('uint8')) / (rows * cols), 4
                    )
                    metadata['source_paths'] = [out_path]
                    _, file_extension = os.path.splitext(out_path)
                    with open(out_path.replace(file_extension, '.json'), 'w') as f:
                        json.dump(metadata, f, indent=4)

                # clean up
                gdal_dataset.FlushCache()
                gdal_dataset = None

        self.logger.info('Processing complete.')

    def _tiff_to_jp2(self, input_tiff_path):
        """
        Convertion of outputs from Geotiff to OpenJPEG
        """

        self.logger.info('--- Converting to JP2OpenJPEG.')

        output_jp2 = input_tiff_path.replace('.tiff', '.jp2')

        src_ds = gdal.Open(input_tiff_path)
        translate_options = gdal.TranslateOptions(
            format='JP2OpenJPEG', creationOptions=['QUALITY=100', 'REVERSIBLE=YES']
        )
        gdal.Translate(output_jp2, src_ds, options=translate_options)
        src_ds = None
        os.remove(input_tiff_path)

    def _run(self):
        """
        Execute the pipeline.
        """
        self._configure()

        # Determine GDAL driver and file extension based on format argument
        format_lower = self._config['output_format'].lower().strip()
        if format_lower in ['tiff', 'tif', 'geotiff']:
            self.driver_name = 'GTiff'
        elif format_lower in ['jp2', 'jpeg2000', 'jpeg 2000']:
            self.driver_name = 'JP2OpenJPEG'
        else:
            raise ValueError(
                f"Unsupported format '{self._config['output_format']}'. Use 'tiff' or 'jp2'."
            )

        if not os.path.exists(self._config['output_folder']):
            os.makedirs(self._config['output_folder'])

        self._parse_metadata()
        self._get_geotransform()
        if self._config['input_water_mask'] is not None:
            self._process_vector_watermask()
        else:
            self._process_watermask()
        self._process_scenes()
