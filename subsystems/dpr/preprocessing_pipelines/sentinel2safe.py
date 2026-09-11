import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path, PosixPath
import numpy
from osgeo import gdal, ogr
from shapely import wkt
from shapely.geometry import mapping
from shapely.ops import transform
from pyproj import Transformer

from .base import PreprocessingBasePipeline
from subsystems.dpr.metadata_processor import MetadataGenerator

# GDAL configuration to handle errors
gdal.UseExceptions()


class Sentinel2SafeProcessor(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-2 SAFE Processor',
        'abstract': 'Perform the clipping and resampling of a Level2A Sentinel-2 SAFE product.'
        'Processing steps include:'
        '(1) Location of the metadata files, extraction of key attributes, list of .jp2 files'
        '(2) Extraction of spectral and SCL bands from the GRANULE. Bands are then cropped and resampled '
        'using GDAL and SCL statistics are recalculated for the new ROI.'
        '(3) Saving output raster (separate bands or multi-band) and metadata to an output folder.',
        'params': {
            'input_safe': {
                'dtype': PosixPath,
                'description': 'Path to the Sentinel-2 unzipped SAFE product',
            },
            'output_folder': {
                'dtype': PosixPath,
                'description': 'Directory where the resulting raster(s) and corresponding metadata will be saved.',
            },
            'output_format': {
                'dtype': str,
                'description': 'Format of the output raster(s): "tiff" or "jp2".',
                'default': 'jp2',
            },
            'overwrite': {
                'dtype': bool,
                'description': 'If true overwrites any raster with same filename present in the output folder.',
                'default': False,
            },
            'roi': {
                'dtype': str,
                'description': 'Bounding box as a POLYGON wkt string, coordinates must be in WGS84 (EPSG:4326)',
            },
            'target_res': {
                'dtype': tuple,
                'description': 'Pixel resolution (x_res, y_res) for the output.',
                'default': (20, 20),
            },
            'resampling_alg': {
                'dtype': str,
                'description': 'The GDAL resampling algorithm (e.g., bilinear, cubic, near).',
                'default': 'near',
            },
            'split_bands': {
                'dtype': bool,
                'description': 'If True, saves each band to a separate raster file. If False, merges all '
                'bands into a single multi-band raster.',
                'default': False,
            },
        },
    }

    def _configure(self):
        """Initialize the processor with None values."""
        self.input_folder = None
        self.s2_metadata = {}

        self.driver_name = None

        # Paths to specific files
        self.msil2a_path = None
        self.mtd_tl_path = None
        self.raster_paths = None

        # Other attributes
        self.granule_list = None
        self.offset_values_list = None
        self.assets = None

    def _locate_metadata_files(self):
        """Locates the required XML files within the input folder hierarchy."""
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if file == 'MTD_MSIL2A.xml':
                    self.msil2a_path = Path(root) / file
                elif file == 'MTD_TL.xml':
                    self.mtd_tl_path = Path(root) / file

        if not self.msil2a_path:
            self.logger.warning('MTD_MSIL2A.xml not found.')
        if not self.mtd_tl_path:
            self.logger.warning('MTD_TL.xml not found.')

        return self.msil2a_path is not None and self.mtd_tl_path is not None

    def _get_clean_tag(self, element):
        """Removes namespace from the tag for easier key mapping."""
        return element.tag.split('}')[-1]

    def _extract_mtd_msil2a(self):
        """Extracts required fields from MTD_MSIL2A.xml."""
        if not self.msil2a_path:
            self.logger.error('MTD_MSIL2A.xml file not found.')
            return

        self.logger.info('Parsing MTD_MSIL2A.xml')
        tree = ET.parse(self.msil2a_path)
        root = tree.getroot()

        # Define tags to find directly (flat search)
        direct_tags = [
            'PRODUCT_TYPE',
            'PRODUCT_DOI',
            'GENERATION_TIME',
            'DATATAKE_TYPE',
            'SENSING_ORBIT_DIRECTION',
        ]

        # Iterative search for basic tags
        for tag in direct_tags:
            elem = root.find(f'.//{tag}')
            if elem is not None:
                self.s2_metadata['properties']['s2:' + tag.lower()] = elem.text

        # Extract Granule_List
        for granule in root.findall('.//Granule'):
            list_images = [img.text for img in granule.findall('IMAGE_FILE')]
            self.granule_list = list_images

        # Extract BOA_ADD_OFFSET_VALUES_LIST
        offsets = {}
        for offset in root.findall('.//BOA_ADD_OFFSET'):
            band_id = offset.get('band_id')
            offsets[f'band_{band_id}'] = offset.text
        self.offset_values_list = offsets

        # Extract Reflectance_Conversion
        ref_conv = {}
        rc_elem = root.find('.//Reflectance_Conversion')
        if rc_elem is not None:
            for child in rc_elem:
                ref_conv[self._get_clean_tag(child)] = child.text
        self.s2_metadata['properties']['s2:reflectance_conversion_factor'] = float(
            ref_conv['U']
        )

        # Extract all physical bands and wavelengths
        wavelengths = {}
        spectral_info = root.findall('.//Spectral_Information')
        for info in spectral_info:
            band_id = info.get('bandId')
            phys_band = info.get('physicalBand')
            w_node = info.find('Wavelength')
            if w_node is not None:
                wavelengths[phys_band] = {
                    'bandId': band_id,
                    'min': float(w_node.findtext('MIN')),
                    'max': float(w_node.findtext('MAX')),
                    'central': float(w_node.findtext('CENTRAL')),
                    'unit': w_node.find('MIN').get('unit')
                    if w_node.find('MIN') is not None
                    else 'nm',
                }

    def _extract_mtd_tl(self):
        """Extracts required fields from MTD_TL.xml (Tile Metadata)."""
        if not self.mtd_tl_path:
            self.logger.error('MTD_TL.xml file not found.')
            return

        self.logger.info('Parsing MTD_TL.xml')
        tree = ET.parse(self.mtd_tl_path)
        root = tree.getroot()

        # CS Name and Code
        cs_name = root.find('.//HORIZONTAL_CS_NAME')
        cs_code = root.find('.//HORIZONTAL_CS_CODE')
        self.s2_metadata['properties']['s2:horizontal_cs_name'] = (
            cs_name.text if cs_name is not None else None
        )
        self.s2_metadata['properties']['s2:horizontal_cs_code'] = (
            cs_code.text if cs_code is not None else None
        )

    def _save_json(self):
        """Save metadata to a JSON file."""
        filename = self.s2_metadata['properties']['s2:product_uri'].replace(
            '.SAFE', '.json'
        )
        output_path = self._config['output_folder'] / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.s2_metadata, f, indent=4)
        self.logger.info(f'Metadata saved to: {output_path}')

    def _roi_transform_wgs_to_utm(self):
        """Transform coordinates from EPSG:4326 to SAFE product UTM coordinates.
        :return: roi transformed  (WKT)
        :rtype: str
        """
        # Convert WKT to Shapely geometry
        geom = wkt.loads(self._config['roi'])

        # Create transformer (From EPSG:4326 to target EPSG)
        target_epsg = self.s2_metadata['properties']['s2:horizontal_cs_code']
        transformer = Transformer.from_crs('EPSG:4326', target_epsg, always_xy=True)

        # Transform the geometry
        transformed_geom = transform(transformer.transform, geom)
        return transformed_geom.wkt

    def _update_metadata_bbox_geometry(self, gdal_dataset):
        """Transform coordinates from EPSG:4326 to SAFE product UTM coordinates.
        :return: roi transformed  (WKT)
        :rtype: str
        """
        gt = gdal_dataset.GetGeoTransform()
        width = gdal_dataset.RasterXSize
        height = gdal_dataset.RasterYSize
        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + gt[1] * width
        ymin = ymax + gt[5] * height
        geom = f'POLYGON (({xmin} {ymin}, {xmin} {ymax}, {xmax} {ymax}, {xmax} {ymin}, {xmin} {ymin}))'
        geom = wkt.loads(geom)
        target_epsg = self.s2_metadata['properties']['s2:horizontal_cs_code']
        transformer = Transformer.from_crs(target_epsg, 'EPSG:4326', always_xy=True)
        transformed_geom = transform(transformer.transform, geom)
        self.s2_metadata['bbox'] = transformed_geom.bounds
        self.s2_metadata['geometry'] = mapping(transformed_geom)

    def _process_scl_statistics(self, slc_array):
        """
        Extracts the SCL band, calculates pixel counts for each class,
        and appends the results to the JSON file.
        """

        if numpy.nanmax(slc_array) > 12 or numpy.nanmax(slc_array) < 0:
            raise ValueError(
                f'Invalid SCL values {numpy.nanmax(slc_array)}. SCL values should be comprised between 1'
                f'and 12.'
            )

        # Get the unique values and their counts
        unique, counts = numpy.unique(slc_array, return_counts=True)
        npix = numpy.sum(counts)

        # List of SLC band labels corresponding to pixel values 0 to 11
        slc_labels = [
            's2:nodata_pixel_percentage',
            's2:saturated_defective_pixel_percentage',
            's2:dark_features_percentage',
            's2:cloud_shadow_percentage',
            's2:vegetation_percentage',
            's2:not_vegetated_percentage',
            's2:water_percentage',
            's2:unclassified_percentage',
            's2:medium_proba_clouds_percentage',
            's2:high_proba_clouds_percentage',
            's2:thin_cirrus_percentage',
            's2:snow_ice_percentage',
        ]

        # Initialize a dictionary with the labels and 0s
        stats_dict = dict.fromkeys(slc_labels, 0)

        # get pixel percentages (float 0.0 - 100.0) for classes appearing in the SCL band
        for k, v in zip(unique, counts):
            stats_dict[slc_labels[k]] = round((v / npix) * 100, 6)

        # Update metadata
        for key in slc_labels:
            self.s2_metadata['properties'][key] = stats_dict[key]

        # Sum of cloud classes (key named 'cloud_cover_pct' as in QCL subsystem)
        sum_cloud_classes = (
            stats_dict['s2:medium_proba_clouds_percentage']
            + stats_dict['s2:high_proba_clouds_percentage']
            + stats_dict['s2:thin_cirrus_percentage']
        )
        self.s2_metadata['properties']['eo:cloud_cover'] = sum_cloud_classes

    def _tiff_to_jp2(self):
        """
        Convertion of outputs from Geotiff to OpenJPEG
        """

        self.logger.info('Converting outputs to JP2OpenJPEG.')

        for input_tiff in self.raster_paths:
            self.logger.info(f'--- {input_tiff}')
            output_jp2 = input_tiff.replace('.tiff', '.jp2')

            src_ds = gdal.Open(input_tiff)
            translate_options = gdal.TranslateOptions(
                format='JP2OpenJPEG', creationOptions=['QUALITY=100', 'REVERSIBLE=YES']
            )
            gdal.Translate(output_jp2, src_ds, options=translate_options)
            src_ds = None
            os.remove(input_tiff)

        self.raster_paths = [
            path.replace('.tiff', '.jp2') for path in self.raster_paths
        ]

        # update assets
        for band_id, properties in self.assets.items():
            # Check if the nested dictionary has an 'href' key
            if 'href' in properties:
                properties['href'] = properties['href'].replace('.tiff', '.jp2')

    def _process_and_merge_jp2(self, roi):
        """
        Makes a list of .jp2 files, resamples them to a consistent resolution,
        crops them to a Region of Interest (ROI), and either merges them into a single GeoTIFF
        or saves each band as a separate GeoTIFF file.

        :param str roi: region of interest
        """

        if self.granule_list is None:
            self.logger.error('GRANULE list is empty. Unable to retrieve .jp2 files.')
            return

        # Band names for output files
        band_names = [
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B09',
            'B11',
            'B12',
            'SCL',
        ]

        # Make an empty dictionary to store the paths to the output rasters (will be used to update the stac assets)
        self.assets = {band: {'href': None} for band in band_names}

        self.logger.info('Converting SAFE .jp2 files to .tiff')
        # Make a list of paths to the jp2 files in the GRANULE folder.
        # For each band uses the file with the smaller pixel size (i.e. 10m if available, else 20m or 60m).
        # Note: make sure that the gdal plugin gdal_JP2OpenJPEG.dll is installed.
        jp2_bands = [
            next((item for item in self.granule_list if string in item), None)
            for string in band_names
        ]
        jp2_paths = [
            os.path.join(self.input_folder, file + '.jp2') for file in jp2_bands
        ]

        # Retrieve the offsets for all spectral bands except band 10
        if self.offset_values_list is not None:
            offsets = [
                int(value)
                for key, value in self.offset_values_list.items()
                if key not in ['band_10']
            ]
            offsets = offsets + [0]  # add 0 for SCL band
        else:
            offsets = [0] * (len(jp2_paths))

        # List resampling algorithms to be used for each spectral bands and add 'near' for SCL band (integers)
        resampling_algorithms = [self._config['resampling_alg']] * (
            len(jp2_paths) - 1
        ) + ['near']

        # Make an empty list to store paths to temp VRT files
        resampled_vrt_files = []

        # Get bounding box
        geom = ogr.CreateGeometryFromWkt(roi)
        xmin, xmax, ymin, ymax = geom.GetEnvelope()
        bounds = [xmin, ymin, xmax, ymax]

        # Resample with gdal.Warp using in-memory VRT files
        for i, (src_file, alg) in enumerate(zip(jp2_paths, resampling_algorithms)):
            vrt_output = os.path.join(
                self._config['output_folder'], f'temp_{i + 1}.vrt'
            )
            self.logger.info(f"--- Resampling {os.path.basename(src_file)} - '{alg}'")
            warp_options = gdal.WarpOptions(
                format='VRT',
                outputType=gdal.GDT_Int16,
                outputBounds=bounds,  # [minX, minY, maxX, maxY] - check projection!
                xRes=self._config['target_res'][0],
                yRes=self._config['target_res'][1],
                resampleAlg=alg,
                multithread=True,  # Speed up processing
                srcNodata=-32768,
                dstNodata=-32768,
            )
            gdal.Warp(vrt_output, src_file, options=warp_options)
            resampled_vrt_files.append(vrt_output)

        base_name = self.s2_metadata['properties']['s2:product_uri'].replace(
            '.SAFE', ''
        )
        if self._config['split_bands']:
            # Save each band as a separate GeoTIFF
            self.logger.info('Saving bands as separate GeoTIFF files')
            output_paths = []

            # Load SCL band for masking
            scl_vrt = resampled_vrt_files[-1]
            scl_ds = gdal.Open(scl_vrt)
            scl_data = scl_ds.GetRasterBand(1)
            mask = scl_data.ReadAsArray() <= 1  # SCL values: NODATA=0 and SATURATED=1
            scl_data.FlushCache()
            # update metadata bbox
            self._update_metadata_bbox_geometry(scl_ds)
            del scl_ds

            for i, (vrt_file, band_name, offset) in enumerate(
                zip(resampled_vrt_files, band_names, offsets)
            ):
                output_filename = f'{base_name}_{band_name}.tiff'
                output_path = os.path.join(
                    self._config['output_folder'], output_filename
                )

                # save filename to assets
                self.assets[band_name]['href'] = os.path.join('./', output_filename)

                # Convert VRT to GeoTIFF or JP2
                options = ['COMPRESS=DEFLATE', 'TILED=YES', 'PREDICTOR=2']
                gdal.Translate(
                    output_path, vrt_file, format='Gtiff', creationOptions=options
                )

                # Apply offset and mask (except for SCL band)
                if band_name != 'SCL':
                    ds = gdal.Open(output_path, gdal.GA_Update)
                    band = ds.GetRasterBand(1)
                    data = band.ReadAsArray()
                    data = data + offset
                    data[mask] = -32768
                    band.WriteArray(data)
                    band.FlushCache()
                    del ds
                else:
                    # get land cover stats from SCL band
                    ds = gdal.Open(output_path, gdal.GA_ReadOnly)
                    band = ds.GetRasterBand(1)
                    scl_data = band.ReadAsArray()
                    self._process_scl_statistics(scl_data)
                    del ds

                output_paths.append(output_path)
                self.logger.info(f'Band {band_name} saved to: {output_path}')

            # Add paths to the metadata
            self.raster_paths = [str(p) for p in output_paths]

        else:
            # merge all VRTs together (gdal.Translate won't accept a list of VRTs)
            mosaic_vrt = os.path.join(
                self._config['output_folder'], 'combined_output.vrt'
            )
            gdal.BuildVRT(mosaic_vrt, resampled_vrt_files, separate=True)

            # Create the final geotiff
            product_name = f'{base_name}.tiff'
            output_path = os.path.join(self._config['output_folder'], product_name)

            # save file name to assets dictionary
            for band in band_names:
                self.assets[band]['href'] = os.path.join('./', product_name)

            options = ['COMPRESS=DEFLATE', 'TILED=YES', 'PREDICTOR=2']
            gdal.Translate(
                output_path, mosaic_vrt, format='GTiff', creationOptions=options
            )

            # Update Geotiff to add bands offsets and use scl band to mask NODATA and SATURATED pixels
            ds = gdal.Open(output_path, gdal.GA_Update)
            scl_band = ds.GetRasterBand(ds.RasterCount)
            mask = scl_band.ReadAsArray() <= 1  # SCL values: NODATA=0 and SATURATED=1
            for i in range(1, ds.RasterCount):
                band = ds.GetRasterBand(i)
                data = band.ReadAsArray()
                data = data + offsets[i - 1]
                data[mask] = -32768
                band.WriteArray(data)
                band.FlushCache()
            # get land cover stats from SCL band
            self._process_scl_statistics(scl_band.ReadAsArray())
            # update metadata bbox
            self._update_metadata_bbox_geometry(ds)
            # Add path to the metadata
            self.raster_paths = [str(output_path)]
            # Clean temp mosaic VRT
            gdal.Unlink(mosaic_vrt)
            del ds
            self.logger.info(f'Raster saved to: {output_path}')

        if self.driver_name == 'JP2OpenJPEG':
            self._tiff_to_jp2()

        # Clean temp files
        for vrt in resampled_vrt_files:
            gdal.Unlink(vrt)

    def _update_stack_assets(self):
        """Append the paths of the generated raster(s) and spectral bands as STAC assets to the metadata."""
        self.s2_metadata['assets'] = {}
        if self.driver_name == 'JP2OpenJPEG':
            asset_type = 'image/jp2'
        else:
            asset_type = 'image/tiff'

        if self._config['split_bands']:
            for band_name, properties in self.assets.items():
                # SCL is categorized with a classification role instead of standard spectral data
                roles = (
                    ['data', 'classification']
                    if band_name == 'SCL'
                    else ['data', 'reflectance']
                )

                asset_entry = {
                    'href': properties['href'].replace('\\', '/'),
                    'type': asset_type,
                    'roles': roles,
                }

                self.s2_metadata['assets'][band_name] = asset_entry

        else:
            # Construct the multiband asset
            self.s2_metadata['assets']['data'] = {
                'href': self.assets['B01']['href'].replace('\\', '/'),
                'type': asset_type,
                'roles': ['data', 'reflectance'],
            }

        # Add SCL band to stac bands
        self.s2_metadata['eo:bands'].append(
            {'name': 'SCL', 'common_name': 'Scene classification map (SCL)', 'gsd': 10}
        )

        # update the stac bands with the spatial resolution of the cropped raster(s)
        for bnd in range(len(self.s2_metadata['eo:bands'])):
            self.s2_metadata['eo:bands'][bnd]['gsd'] = int(
                numpy.abs(self._config['target_res'][0])
            )

    def _run(self):
        self._configure()
        # Ensure paths are Path objects
        if isinstance(self._config['output_folder'], Path) is False:
            self._config['output_folder'] = Path(self._config['output_folder'])
        if isinstance(self._config['input_safe'], Path) is False:
            self._config['input_safe'] = Path(self._config['input_safe'])

        # Ensure output folder exists
        self._config['output_folder'].mkdir(parents=True, exist_ok=True)

        product = os.path.basename(self._config['input_safe'])
        if self._config['overwrite'] is False:
            path = os.path.join(
                self._config['output_folder'], product.replace('.zip', '.tiff')
            )
            if (
                os.path.exists(path)
                or os.path.exists(path.replace('.tiff', '.jp2'))
                or os.path.exists(path.replace('.tiff', '_B01.tiff'))
                or os.path.exists(path.replace('.tiff', '_B01.jp2'))
            ):
                self.logger.info(
                    f'Skipping processing of {product}: '
                    f'A raster file with same name already exists in the output folder.'
                )
                return

        self.logger.info(f'Processing {product}')

        # Determine GDAL driver and file extension based on format argument
        format_lower = self._config['output_format'].lower().strip()
        if format_lower in ['tiff', 'tif', 'geotiff']:
            self.driver_name = 'GTiff'
        elif format_lower in ['jp2', 'jpeg2000']:
            self.driver_name = 'JP2OpenJPEG'
        else:
            raise ValueError(
                f"Unsupported format '{self._config['output_format']}'. Use 'tiff' or 'jp2'."
            )

        self.input_folder = self._config['input_safe']

        self.logger.info(f'Processing folder: {self.input_folder}')
        mdgen = MetadataGenerator()
        mdgen.set_datasource(self.input_folder)
        self.s2_metadata = mdgen.stac.create_item()

        # Proceed with the conversion of the SAFE to Geotiff
        if self._locate_metadata_files():
            self._extract_mtd_msil2a()
            self._extract_mtd_tl()
            roi = self._roi_transform_wgs_to_utm()
            self._process_and_merge_jp2(roi)
            self._update_stack_assets()
            self._save_json()
        else:
            self.logger.error(
                'Extraction failed: Required metadata files could not be located.'
            )
