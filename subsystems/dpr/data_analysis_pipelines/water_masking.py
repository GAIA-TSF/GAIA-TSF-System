import os
import json
import glob
from pathlib import PosixPath
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, osr, ogr
from scipy.ndimage import label

from .base import DataAnalysisBasePipeline


class Sentinel2WaterMaskingPipeline(DataAnalysisBasePipeline):
    metadata = {
        'title': 'Sentinel-2 Water Masking',
        'abstract': 'TBD',
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
                               'instead of relying on the pipeline s filtering.',
                'default': None,
            },
            'input_months': {
                'dtype': list,
                'description': 'OPTIONAL. List of months (e.g., [1, 2, 5, 6]) to use for water masking. Only applies'
                               'if no vector file is provided by the "input_water_mask" parameter. ',
                'default': None,
            },
            'start_date': {
                'dtype': str,
                'description': 'OPTIONAL. Starting date for temporal filtering as "YYYY-MM-DD".',
                'default': None,
            },
            'end_date': {
                'dtype': str,
                'description': 'OPTIONAL. End date for temporal filtering as "YYYY-MM-DD".',
                'default': None,
            },
        },
    }

    def _configure(self):
        """Initialize the processor with None values."""
        self.global_watermask = None
        self.scenes_metadata = None
        self.scenes_metadata_filtered = None

        # Sentinel-2 reference spectra for SAM (11 bands, band 9 is excluded)
        self.ref_soil = np.array([261., 498.5, 748.5, 971., 1385., 2026., 2310., 2550., 2691.5, 2933., 1845.])
        self.ref_veg = np.array([32., 124., 285., 183., 531., 1603., 1903.5, 2018., 2081., 921., 432.])

    def _parse_metadata(self):
        """
        Retrieve metadata from json files and store them to a dictionary ('scenes_metadata'). Also perform a filtering
        ('scenes_metadata_filtered') based on the maximum percentage of clouds/snow/dark pixels (max_cloud_snow_dark
        parameter).
        """
        records = []
        json_files = glob.glob(os.path.join(self._config['input_folder'], "**/*.json"), recursive=True)
        print(f"Parsing metadata. Input folder: {self._config['input_folder']}.")
        print(f"The input folder contains {len(json_files)} entries.")

        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                scl_stats = data.get("SCL_classes_pct", {})
                record = {
                    "DATATAKE_SENSING_START": data.get("DATATAKE_SENSING_START"),
                    "source_path": data.get("source_path"),
                    "SCL_snow_or_ice": scl_stats.get("SCL_snow_or_ice", 0),
                    "cloud_cover_pct": data.get("cloud_cover_pct", 0),
                    "SCL_cloud_shadows": scl_stats.get("SCL_cloud_shadows", 0),
                    "SCL_dark_areas": scl_stats.get("SCL_dark_areas", 0)
                }
                records.append(record)
            except Exception as e:
                print(f"Error parsing {json_path}: {e}")

        self.scenes_metadata = pd.DataFrame(records)

        if not self.scenes_metadata.empty:
            # Temporal filtering
            if self._config['start_date'] is not None or self._config['end_date'] is not None:
                # Ensure we are working with a copy to avoid SettingWithCopy warnings
                df = self.scenes_metadata.copy()

                # Convert column to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df['DATATAKE_SENSING_START']):
                    df['DATATAKE_SENSING_START'] = pd.to_datetime(df['DATATAKE_SENSING_START'])

                # Apply start date filter
                if self._config['start_date'] is not None:
                    if pd.to_datetime(self._config['start_date']).tz is None:
                        start_dt = pd.to_datetime(self._config['start_date']).tz_localize('UTC')
                    else:
                        start_dt = pd.to_datetime(self._config['start_date'])
                    df = df[df['DATATAKE_SENSING_START'] >= start_dt]
                    print(f"Applied start date filter: {start_dt}")

                # Apply end date filter
                if self._config['end_date'] is not None:
                    if pd.to_datetime(self._config['end_date']).tz is None:
                        end_dt = pd.to_datetime(self._config['end_date']).tz_localize('UTC')
                    else:
                        end_dt = pd.to_datetime(self._config['end_date'])
                    df = df[df['DATATAKE_SENSING_START'] <= end_dt]
                    print(f"Applied end date filter: {end_dt}")

                # Update the filtered metadata attribute
                self.scenes_metadata = df.reset_index(drop=True)
            print(f"Temporal filtering. {len(self.scenes_metadata)} entries remains.")

            # Cloud/Snow/Dark pixels filtering
            self.scenes_metadata["cloud_snow_dark"] = (
                    self.scenes_metadata["SCL_dark_areas"] +
                    self.scenes_metadata["SCL_snow_or_ice"] +
                    self.scenes_metadata["SCL_cloud_shadows"] +
                    self.scenes_metadata["cloud_cover_pct"]
            )

            self.scenes_metadata_filtered = self.scenes_metadata[
                self.scenes_metadata["cloud_snow_dark"] < self._config['max_cloud_snow_dark']
                ].copy()

            self.scenes_metadata_filtered['DATATAKE_SENSING_START'] = pd.to_datetime(
                self.scenes_metadata_filtered['DATATAKE_SENSING_START'])
            self.scenes_metadata_filtered = self.scenes_metadata_filtered.sort_values(
                "DATATAKE_SENSING_START").reset_index(drop=True)

            print(f"Cloud/Snow/Dark pixels filtering. {len(self.scenes_metadata_filtered)} entries remains.")

        print(f"Metadata parsed.")

    def _get_geotransform(self):
        """
        Retrieve spatial data from first GeoTIFF in the input folder. Assumes that all scenes in the input folder share
        the same EPSG and geotransform. Only used if a vector file is provided (input_water_mask is not None).
        """
        tiff_files = glob.glob(os.path.join(self._config['input_folder'], "*.tif*"))
        if tiff_files:
            ds = gdal.Open(tiff_files[0])
            if ds:
                self.geotransform = ds.GetGeoTransform()
                self.projection = ds.GetProjection()
                self.shape = (ds.RasterYSize, ds.RasterXSize)

                # Get EPSG
                srs = osr.SpatialReference(wkt=self.projection)
                self.epsg = srs.GetAttrValue("AUTHORITY", 1)

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
        if not self.bounds:
            raise ValueError("Could not retrieve the bounding box.")

        # Load vector
        gdf = gpd.read_file(self._config['input_water_mask'])

        # Reproject if necessary
        target_crs = f"EPSG:{self.epsg}"
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Crop to Bounding Box
        from shapely.geometry import box as sbox
        bbox_geom = sbox(*self.bounds)
        gdf = gdf.clip(bbox_geom)

        if gdf.empty:
            raise ValueError("Could not parse the input water mask vectors.")

        # Optional Buffer
        if buffer_m != 0:
            gdf['geometry'] = gdf.buffer(buffer_m)

        # Rasterize and Label by Area (1 = largest)
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
        gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=rank"])

        self.global_watermask = target_ds.GetRasterBand(1).ReadAsArray()

        # Cleanup
        target_ds = None
        mem_ds = None

    def _process_watermask(self):
        """
        Creates a median water mask by processing scenes listed in scenes_metadata_filtered.
        """

        if self.scenes_metadata_filtered is None or self.scenes_metadata_filtered.empty:
            print("No filtered scenes available to compute median water mask.")
            return

        df = self.scenes_metadata_filtered.copy()

        # Filter by months if requested
        if self._config['input_months']:
            # Ensure DATATAKE_SENSING_START is datetime (already handled in _parse_metadata, but being safe)
            df['month'] = df['DATATAKE_SENSING_START'].dt.month
            df = df[df['month'].isin(self._config['input_months'])]
            print(f"Filtered {len(df)} scenes matching months: {self._config['input_months']}")

            if df.empty:
                raise ValueError("No scenes found for the selected month range.")

        print(f"Computing median water mask from {len(df)} scenes...")

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
            raise ValueError("Failed to read data from any scenes for water mask generation.")

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

        # Prepare data for SAM (Sentinel-2's Band 9 is excluded)
        sam_bands_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
        sam_soil = self._calculate_spectral_angle(self.ref_soil, median_ds, sam_bands_idx)
        sam_veg = self._calculate_spectral_angle(self.ref_veg, median_ds, sam_bands_idx)

        # Generate binary mask based on thresholds
        binary_mask = (
                (slope < 1) & (intercept > -400) &
                (si > 900) & (sam_soil > 0.15) & (sam_veg > 0.15) &
                (swir_reflect < 1000)
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

        print("Median water mask generated successfully.")

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
        norm_product = np.linalg.norm(spectra_flat, axis=1) * np.linalg.norm(reference_spectrum)

        # Prevent division by zero and handle out-of-range for arccos
        cosine_sim = np.divide(dot_product, norm_product, out=np.zeros_like(dot_product), where=norm_product != 0)
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

        return np.nan_to_num(slope_map, nan=-9999), np.nan_to_num(intercept_map, nan=-9999)

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
        swir_reflect = (dataset.GetRasterBand(11).ReadAsArray() + dataset.GetRasterBand(12).ReadAsArray()) / 2
        return swir_reflect

    def _process_scenes(self):
        """
        Iterates through GeoTIFFs, applies spectral refinement to the base mask,
        and exports a multi-band result.
        """

        tiffs = list(self.scenes_metadata_filtered["source_path"])

        for tiff_path in tiffs:
            print(tiff_path)

            gdal_dataset = gdal.Open(tiff_path)
            if gdal_dataset is None:
                return None

            si = self._calculate_shadow_index(gdal_dataset)
            slope, intercept = self._calculate_vectorized_regression(gdal_dataset)
            swir_reflect = self._get_swir_reflectance(gdal_dataset)

            # Prepare data for SAM (Sentinel-2's Band 9 is excluded)
            sam_bands_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
            sam_soil = self._calculate_spectral_angle(self.ref_soil, gdal_dataset, sam_bands_idx)
            sam_veg = self._calculate_spectral_angle(self.ref_veg, gdal_dataset, sam_bands_idx)

            # Spectral refinement condition (pixels to be REMOVED from the global water mask)
            remove_conditions = (si < 900) | (slope > 1) | (intercept < -500) | (sam_soil < 0.15) | (sam_veg < 0.15) | \
                                (swir_reflect > 1000)

            refined_mask = self.global_watermask.copy()
            refined_mask[remove_conditions] = 0

            if np.nansum(refined_mask) > 0:
                # Export
                out_name = os.path.basename(tiff_path)
                out_path = os.path.join(self._config['output_folder'], out_name)

                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(out_path, gdal_dataset.RasterXSize, gdal_dataset.RasterYSize,
                                       gdal_dataset.RasterCount + 1, gdal.GDT_UInt16)
                out_ds.SetGeoTransform(gdal_dataset.GetGeoTransform())
                out_ds.SetProjection(gdal_dataset.GetProjection())

                # Copy original bands
                for i in range(1, gdal_dataset.RasterCount + 1):
                    out_ds.GetRasterBand(i).WriteArray(gdal_dataset.GetRasterBand(i).ReadAsArray())

                # Add updated mask as last band
                out_ds.GetRasterBand(gdal_dataset.RasterCount + 1).WriteArray(refined_mask)
                out_ds.FlushCache()
                out_ds = None
                ds = None

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
