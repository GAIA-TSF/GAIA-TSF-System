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
        '(3) Generation of a water mask for each Sentinel-2 scenes. Thresholding on spectral indices'
        '    (similar to step 2B) will be used to check the global water mask validity for a given scene.'
        '    Pixels that do not fulfill thresholding conditions are removed to create a scene mask.'
        '    If the scene mask contains valid pixels, it will be appended to the Sentinel-2 bands and the'
        '    merged raster will be exported to the output folder.',
        'params': {
            'input_folder': {
                'dtype': PosixPath,
                'description': 'Path to the folder containing the processed Sentinel-2 SAFE products (geotiffs)',
            },
            'output_folder': {
                'dtype': PosixPath,
                'description': 'Directory where the processed geotiffs will be saved.',
            },
            'max_cloud_snow_dark': {
                'dtype': float,
                'description': 'Parameter used to filter scenes based on the maximum percentage of pixels containing '
                'clouds/snow/shadows. Ratio between 0 and 1. Scenes with a ratio above this value will '
                'be filtered out',
                'default': 0.1,
            },
            'input_water_mask': {
                'dtype': PosixPath,
                'description': 'OPTIONAL. The user can provide a vector file (e.g., .shp, .gpkg) for water bodies '
                'instead of relying on the pipeline filtering.',
                'default': 'None',
            },
            'input_months': {
                'dtype': list,
                'description': 'OPTIONAL. List of months (e.g., [1, 2, 5, 6]) to use for water masking. Only applies'
                'if no vector file is provided by the "input_water_mask" parameter. ',
                'default': [],
            },
            'start_date': {
                'dtype': str,
                'description': 'OPTIONAL. Starting date for temporal filtering as "YYYY-MM-DD".',
                'default': 'None',
            },
            'end_date': {
                'dtype': str,
                'description': 'OPTIONAL. End date for temporal filtering as "YYYY-MM-DD".',
                'default': 'None',
            },
        },
    }

    def _configure(self):
        """Initialize the processor with None values."""
        if self._config['input_water_mask'] == 'None':
            self._config['input_water_mask'] = None

        if self._config['start_date'] == 'None':
            self._config['start_date'] = None

        if self._config['end_date'] == 'None':
            self._config['end_date'] = None

        if not self._config['input_months']:
            self._config['input_months'] = None

        self.global_watermask = None
        self.scenes_metadata = None
        self.scenes_metadata_filtered = None

        # Sentinel-2 reference spectra for SAM (11 bands, band 9 is excluded)
        self.ref_soil = np.array(
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
            ]
        )
        self.ref_veg = np.array(
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
            ]
        )

    def _parse_metadata(self):
        """
        Retrieve metadata from json files and store them to a dictionary ('scenes_metadata'). Also perform a filtering
        ('scenes_metadata_filtered') based on the maximum percentage of clouds/snow/dark pixels (max_cloud_snow_dark
        parameter).
        """
        self.logger.info(
            f'Parsing metadata. Input folder: {self._config["input_folder"]}.'
        )

        try:
            json_files = glob.glob(
                os.path.join(self._config['input_folder'], '**/*.json'), recursive=True
            )
        except Exception as e:
            self.logger.error(f'Error parsing {self._config["input_folder"]}: {e}')

        self.logger.info(f'--- The input folder contains {len(json_files)} entries.')

        records = []
        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                scl_stats = data.get('SCL_classes_pct', {})
                record = {
                    'DATATAKE_SENSING_START': data.get('DATATAKE_SENSING_START'),
                    'source_path': data.get('source_path'),
                    'SCL_snow_or_ice': scl_stats.get('SCL_snow_or_ice', 0),
                    'cloud_cover_pct': data.get('cloud_cover_pct', 0),
                    'SCL_cloud_shadows': scl_stats.get('SCL_cloud_shadows', 0),
                    'SCL_dark_areas': scl_stats.get('SCL_dark_areas', 0),
                }
                records.append(record)
            except Exception as e:
                self.logger.warning(f'Error parsing {json_path}: {e}')

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
        Retrieve spatial data from first GeoTIFF in the input folder. Assumes that all scenes in the input folder share
        the same EPSG and geotransform. Only used if a vector file is provided (input_water_mask is not None).
        """
        self.logger.info('Retrieving spatial data from GeoTIFFs in input folder')
        try:
            tiff_files = glob.glob(os.path.join(self._config['input_folder'], '*.tif*'))
        except Exception as e:
            self.logger.warning(f'Error locating GeoTIFFs: {e}')

        if tiff_files:
            ds = gdal.Open(tiff_files[0])
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

    def _process_vector_watermask(self, buffer_m=0):
        """
        Ingests vector, reprojects, crops to bounds, and creates a size-labeled binary mask.
        """

        self.logger.info('Generating global water mask from vector file.')

        if not self.bounds:
            self.logger.error('Bounding box is missing.')

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
            self.logger.error('Could not parse the input water mask vectors.')

        # Optional Buffer
        if buffer_m != 0:
            gdf['geometry'] = gdf.buffer(buffer_m)

        # Rasterize and Label by area (1 = largest)
        gdf['area'] = gdf.geometry.area

        # Sort descending by area so the first elements are the largest
        gdf = gdf.sort_values(by='area', ascending=False).reset_index(drop=True)

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

        # Add a field for the label/rank
        field_defn = ogr.FieldDefn('rank', ogr.OFTInteger)
        layer.CreateField(field_defn)

        # Populate the memory layer with features and their rank (1-based)
        for i, row in gdf.iterrows():
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(ogr.CreateGeometryFromWkb(row.geometry.wkb))
            feature.SetField('rank', int(i + 1))
            layer.CreateFeature(feature)
            feature = None

        # Rasterize using the 'rank' attribute
        gdal.RasterizeLayer(target_ds, [1], layer, options=['ATTRIBUTE=rank'])

        self.global_watermask = target_ds.GetRasterBand(1).ReadAsArray()

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
            ds = gdal.Open(row['source_path'])
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
        first_tiff = df.iloc[0]['source_path']
        ds_ref = gdal.Open(first_tiff)
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

        # Calculate spectral indices
        si = self._calculate_shadow_index(median_ds)
        slope, intercept = self._calculate_vectorized_regression(median_ds)
        swir_reflect = self._get_swir_reflectance(median_ds)
        band1 = median_ds.GetRasterBand(1).ReadAsArray().astype(np.int16)

        # Prepare data for SAM (Sentinel-2's Band 9 is excluded)
        sam_bands_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
        sam_soil = self._calculate_spectral_angle(
            self.ref_soil, median_ds, sam_bands_idx
        )
        sam_veg = self._calculate_spectral_angle(self.ref_veg, median_ds, sam_bands_idx)

        # Generate binary mask based on thresholds
        binary_mask = (
            (slope < 1)
            & (intercept > -400)
            & (si > 900)
            & (sam_soil > 0.15)
            & (sam_veg > 0.15)
            & (swir_reflect < 1000)
            & (band1 < 2000)
        ).astype(np.uint8)

        # Label components (we want the largest body to be 1)
        # Using scipy.ndimage.label
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

            self.global_watermask = ranked_mask
        else:
            self.global_watermask = binary_mask

        self.logger.info('Global water mask generated successfully.')

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
        denominator = np.sum(x_diff**2)

        m = numerator / denominator
        b = y_mean - (m * x_mean)

        slope_map = m.reshape((n_rows, n_cols))
        intercept_map = b.reshape((n_rows, n_cols))

        return np.nan_to_num(slope_map, nan=-9999), np.nan_to_num(
            intercept_map, nan=-9999
        )

    def _calculate_shadow_index(self, dataset):
        """
        Calculates the shadow index (NIR bands) for a gdal dataset.
        """
        indices = [6, 7, 8, 9]
        bands = []
        for i in indices:
            b = dataset.GetRasterBand(i).ReadAsArray().astype(np.float32)
            # Simple normalization (assuming 10000 reflectance scale)
            bands.append(b / 10000.0)

        term = (1.0 - bands[0]) * (1.0 - bands[1]) * (1.0 - bands[2]) * (1.0 - bands[3])
        term = np.maximum(term, 0)
        si = (term ** (1.0 / 10.0) * 1000).astype('uint16')
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

    def _process_scenes(self):
        """
        Iterates through GeoTIFFs, applies spectral refinement to the base mask,
        and exports a multi-band result.
        """
        self.logger.info('Generating water masks for all scenes.')
        self.logger.info(
            f'Merged raster will be exported to {self._config["output_folder"]}'
        )

        if (
            self._config['start_date'] is not None
            or self._config['end_date'] is not None
        ):
            self.logger.info(f'Temporal filter start = {self._config["start_date"]}')
            self.logger.info(f'Temporal filter end = {self._config["end_date"]}')

        tiffs = list(self.scenes_metadata['source_path'])

        for tiff_path in tiffs:
            self.logger.info(f'--- Processing {os.path.basename(tiff_path)}')

            gdal_dataset = gdal.Open(tiff_path)
            if gdal_dataset is None:
                return None

            si = self._calculate_shadow_index(gdal_dataset)
            slope, intercept = self._calculate_vectorized_regression(gdal_dataset)
            swir_reflect = self._get_swir_reflectance(gdal_dataset)

            # Prepare data for SAM (Sentinel-2's Band 9 is excluded)
            sam_bands_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
            sam_soil = self._calculate_spectral_angle(
                self.ref_soil, gdal_dataset, sam_bands_idx
            )
            sam_veg = self._calculate_spectral_angle(
                self.ref_veg, gdal_dataset, sam_bands_idx
            )
            band1 = gdal_dataset.GetRasterBand(1).ReadAsArray().astype(np.int16)

            # Spectral refinement condition (pixels to be REMOVED from the global water mask)
            remove_conditions = (
                (si < 900)
                | (slope > 1)
                | (intercept < -500)
                | (sam_soil < 0.15)
                | (sam_veg < 0.15)
                | (swir_reflect > 1000)
                | (band1 > 2000)
            )

            refined_mask = self.global_watermask.copy()
            refined_mask[remove_conditions] = 0

            if np.nansum(refined_mask) > 0:
                # Export if mask contains valid pixels
                out_name = os.path.basename(tiff_path)
                out_path = os.path.join(self._config['output_folder'], out_name)

                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(
                    out_path,
                    gdal_dataset.RasterXSize,
                    gdal_dataset.RasterYSize,
                    gdal_dataset.RasterCount + 1,
                    gdal.GDT_UInt16,
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

                # Copy and update metadata
                json_path = tiff_path.replace('.tiff', '.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    rows, cols = gdal_dataset.RasterYSize, gdal_dataset.RasterXSize
                    metadata['water_mask_pct'] = round(
                        np.nansum((refined_mask > 0).astype('uint8')) / (rows * cols), 4
                    )
                    with open(out_path.replace('.tiff', '.json'), 'w') as f:
                        json.dump(metadata, f, indent=4)

        self.logger.info('Processing complete.')

    def _run(self):
        """
        Execute the pipeline.
        """
        self._configure()

        if not os.path.exists(self._config['output_folder']):
            os.makedirs(self._config['output_folder'])

        self._parse_metadata()
        self._get_geotransform()
        if self._config['input_water_mask'] is not None:
            self._process_vector_watermask()
        else:
            self._process_watermask()
        self._process_scenes()
