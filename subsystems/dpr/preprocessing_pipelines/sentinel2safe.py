import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path, PosixPath

from osgeo import gdal, ogr
from shapely import wkt
from shapely.ops import transform
from pyproj import Transformer

from .base import PreprocessingBasePipeline

# GDAL configuration to handle errors
gdal.UseExceptions()


class Sentinel2SafeProcessor(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-2 SAFE Processor',
        'abstract': 'Perform the conversion of a Level2A Sentinel-2 SAFE product to a geotiff.'
        'Processing steps include:'
        '(1) Location of the metadata files, extraction of key attributes, list of .jp2 files'
        '(2) Extraction of spectral and SCL bands from the GRANULE. Bands are then cropped and resampled '
        'using GDAL'
        '(3) Saving output geotiff and metadata to an output folder.',
        'params': {
            'input_safe': {
                'dtype': PosixPath,
                'description': 'Path to the Sentinel-2 SAFE product (can be a .zip file or an unzipped folder)',
            },
            'output_folder': {
                'dtype': PosixPath,
                'description': 'Directory where the resulting geotiff and corresponding metadata will be saved.',
            },
            'overwrite': {
                'dtype': bool,
                'description': 'If true overwrites any geotiff with same filename present in the output folder.',
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
                'description': 'If True, saves each band to a separate GeoTIFF file. If False, merges all bands into a single GeoTIFF.',
                'default': False,
            },
        },
    }

    def _configure(self):
        """Initialize the processor with None values."""
        self.input_folder = None
        self.s2_metadata = {}

        # Paths to specific files
        self.msil2a_path = None
        self.mtd_tl_path = None

        # Other attributes
        self.granule_list = None
        self.offset_values_list = None

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
            'PRODUCT_START_TIME',
            'PRODUCT_STOP_TIME',
            'PRODUCT_URI',
            'PROCESSING_LEVEL',
            'PRODUCT_TYPE',
            'PRODUCT_DOI',
            'GENERATION_TIME',
            'SPACECRAFT_NAME',
            'DATATAKE_SENSING_START',
            'DATATAKE_TYPE',
            'SENSING_ORBIT_NUMBER',
            'SENSING_ORBIT_DIRECTION',
        ]

        # Iterative search for basic tags
        for tag in direct_tags:
            elem = root.find(f'.//{tag}')
            if elem is not None:
                self.s2_metadata[tag] = elem.text

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
        self.s2_metadata['Reflectance_Conversion'] = ref_conv

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
                    'min': w_node.findtext('MIN'),
                    'max': w_node.findtext('MAX'),
                    'central': w_node.findtext('CENTRAL'),
                    'unit': w_node.find('MIN').get('unit')
                    if w_node.find('MIN') is not None
                    else 'nm',
                }
        self.s2_metadata['Wavelengths'] = wavelengths

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
        self.s2_metadata['HORIZONTAL_CS_NAME'] = (
            cs_name.text if cs_name is not None else None
        )
        self.s2_metadata['HORIZONTAL_CS_CODE'] = (
            cs_code.text if cs_code is not None else None
        )

    def _save_json(self):
        """Save metadata to a JSON file."""
        filename = self.s2_metadata['PRODUCT_URI'].replace('.SAFE', '.json')
        output_path = self._config['output_folder'] / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.s2_metadata, f, indent=4)
        self.logger.info(f'Metadata saved to: {output_path}')

    def _roi_transform(self):
        """Transform coordinates from EPSG:4326 to SAFE product coordinates.

        :return roi transformed  (WKT)
        :rtype: str
        """
        # Convert WKT to Shapely geometry
        geom = wkt.loads(self._config['roi'])

        # Create transformer (From EPSG:4326 to target EPSG)
        target_epsg = self.s2_metadata['HORIZONTAL_CS_CODE']
        transformer = Transformer.from_crs('EPSG:4326', target_epsg, always_xy=True)

        # Transform the geometry
        transformed_geom = transform(transformer.transform, geom)
        return transformed_geom.wkt

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

        self.logger.info('Converting .jp2 files to .tiff')
        # Make a list of paths to the jp2 files in the GRANULE folder.
        # For each band uses the file with the smaller pixel size (i.e. 10m if available, else 20m or 60m).
        # Note: make sure that the gdal plugin gdal_JP2OpenJPEG.dll is installed.
        jp2_bands = [
            next((item for item in self.granule_list if string in item), None)
            for string in [
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

        # Resample with gdal.Warp using in-memory VRT files
        geom = ogr.CreateGeometryFromWkt(roi)
        xmin, xmax, ymin, ymax = geom.GetEnvelope()
        bounds = [xmin, ymin, xmax, ymax]
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

        if self._config['split_bands']:
            # Save each band as a separate GeoTIFF
            self.logger.info('Saving bands as separate GeoTIFF files')
            output_paths = []
            base_name = self.s2_metadata['PRODUCT_URI'].replace('.SAFE', '')

            # Load SCL band for masking
            scl_vrt = resampled_vrt_files[-1]
            scl_ds = gdal.Open(scl_vrt)
            scl_data = scl_ds.GetRasterBand(1).ReadAsArray()
            mask = scl_data <= 1  # SCL values: NODATA=0 and SATURATED=1
            del scl_ds

            for i, (vrt_file, band_name, offset) in enumerate(
                zip(resampled_vrt_files, band_names, offsets)
            ):
                output_filename = f'{base_name}_{band_name}.tiff'
                output_path = os.path.join(
                    self._config['output_folder'], output_filename
                )
                options = ['COMPRESS=LZW', 'TILED=YES']

                # Convert VRT to GeoTIFF
                gdal.Translate(
                    output_path, vrt_file, format='GTiff', creationOptions=options
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

                output_paths.append(output_path)
                self.logger.info(f'Band {band_name} saved to: {output_path}')

            # Add paths to the metadata
            self.s2_metadata['source_paths'] = [str(p) for p in output_paths]
        else:
            # merge all VRTs together (gdal.Translate won't accept a list of VRTs)
            mosaic_vrt = os.path.join(
                self._config['output_folder'], 'combined_output.vrt'
            )
            gdal.BuildVRT(mosaic_vrt, resampled_vrt_files, separate=True)

            # Create the final geotiff
            product_name = self.s2_metadata['PRODUCT_URI'].replace('.SAFE', '.tiff')
            output_path = os.path.join(self._config['output_folder'], product_name)
            options = ['COMPRESS=LZW', 'TILED=YES']
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
            del ds
            self.logger.info(f'GeoTIFF saved to: {output_path}')

            # Add path to the metadata
            self.s2_metadata['source_path'] = str(output_path)

            # Clean temp mosaic VRT
            gdal.Unlink(mosaic_vrt)

        # Clean temp files
        for vrt in resampled_vrt_files:
            gdal.Unlink(vrt)

    def _run(self):
        # Ensure output folder exists
        self._config['output_folder'].mkdir(parents=True, exist_ok=True)

        product = os.path.basename(self._config['input_safe'])
        self.logger.info('Processing {product}')
        path = os.path.join(
            self._config['output_folder'], product.replace('.zip', '.tiff')
        )
        if os.path.exists(path) and self._config['overwrite'] is False:
            self.logger.info(
                'Skipping processing: A geotiff file already exists in the output folder.'
            )
            return

        # check if input safe is a folder or a zip. Unzip to a temporary folder if necessary.
        self.input_folder = self._config['input_safe']

        # Proceed with the conversion of the SAFE to Geotiff
        self.logger.info(f'Scanning folder: {self.input_folder}')
        if self._locate_metadata_files():
            self.s2_metadata['Input_SAFE_path'] = str(self._config['input_safe'])
            self._extract_mtd_msil2a()
            self._extract_mtd_tl()
            roi = self._roi_transform()
            self._process_and_merge_jp2(roi)
            self._save_json()
        else:
            self.logger.error(
                'Extraction failed: Required metadata files could not be located.'
            )
