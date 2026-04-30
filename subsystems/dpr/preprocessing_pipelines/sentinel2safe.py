from .base import BasePipeline
import os
import json
import zipfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from osgeo import gdal, ogr
from shapely import wkt
from shapely.ops import transform
from pyproj import Transformer

# GDAL configuration to handle errors
gdal.UseExceptions()


class Sentinel2SafeProcessor(BasePipeline):
    metadata = {
        'title': 'Sentinel-2 SAFE Processor',
        'abstract': 'Perform the conversion of a Level2A Sentinel-2 SAFE product to a geotiff.'
        'Processing steps include:'
        '(1) Location of the metadata files, extraction of key attributes, list of .jp2 files'
        '(2) Extraction of spectral and SCL bands from the GRANULE. Bands are then cropped and resampled '
        'using GDAL'
        '(3) Saving output geotiff and metadata to an output folder.',
    }

    def __init__(self):
        """Initialize the processor with None values."""
        self.input_folder = None
        self.output_folder = None
        self.roi = None
        self.target_res = None
        self.resampling_alg = None
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
            print('Warning: MTD_MSIL2A.xml not found.')
        if not self.mtd_tl_path:
            print('Warning: MTD_TL.xml not found.')

        return self.msil2a_path is not None and self.mtd_tl_path is not None

    def _get_clean_tag(self, element):
        """Removes namespace from the tag for easier key mapping."""
        return element.tag.split('}')[-1]

    def _extract_mtd_msil2a(self):
        """Extracts required fields from MTD_MSIL2A.xml."""
        if not self.msil2a_path:
            print('MTD_MSIL2A.xml file not found.')
            return

        print('Parsing MTD_MSIL2A.xml')
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
            print('MTD_TL.xml file not found.')
            return

        print('Parsing MTD_TL.xml')
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
        output_path = self.output_folder / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.s2_metadata, f, indent=4)
        print(f'Metadata saved to: {output_path}')

    def _roi_transform(self):
        """Transform coordinates from EPSG:4326 to SAFE product coordinates."""
        # Convert WKT to Shapely geometry
        geom = wkt.loads(self.roi)

        # Create transformer (From EPSG:4326 to target EPSG)
        target_epsg = self.s2_metadata['HORIZONTAL_CS_CODE']
        transformer = Transformer.from_crs('EPSG:4326', target_epsg, always_xy=True)

        # Transform the geometry
        transformed_geom = transform(transformer.transform, geom)
        self.roi = transformed_geom.wkt

    def _process_and_merge_jp2(self):
        """
        Makes a list of .jp2 files, resamples them to a consistent resolution,
        crops them to a Region of Interest (ROI), and merges them into a single GeoTIFF.
        """

        if self.granule_list is None:
            print('GRANULE list is empty. Unable to retrieve .jp2 files.')
            return

        print('Converting .jp2 files to .tiff')
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
        resampling_algorithms = [self.resampling_alg] * (len(jp2_paths) - 1) + ['near']

        # Make an empty list to store paths to temp VRT files
        resampled_vrt_files = []

        # Resample with gdal.Warp using in-memory VRT files
        geom = ogr.CreateGeometryFromWkt(self.roi)
        xmin, xmax, ymin, ymax = geom.GetEnvelope()
        bounds = [xmin, ymin, xmax, ymax]
        for i, (src_file, alg) in enumerate(zip(jp2_paths, resampling_algorithms)):
            vrt_output = os.path.join(self.output_folder, f'temp_{i + 1}.vrt')
            print(f"--- Resampling {os.path.basename(src_file)} - '{alg}'")
            warp_options = gdal.WarpOptions(
                format='VRT',
                outputType=gdal.GDT_Int16,
                outputBounds=bounds,  # [minX, minY, maxX, maxY] - check projection!
                xRes=self.target_res[0],
                yRes=self.target_res[1],
                resampleAlg=alg,
                multithread=True,  # Speed up processing
                srcNodata=-32768,
                dstNodata=-32768,
            )
            gdal.Warp(vrt_output, src_file, options=warp_options)
            resampled_vrt_files.append(vrt_output)

        # merge all VRTs together (gdal.Translate won't accept a list of VRTs)
        mosaic_vrt = os.path.join(self.output_folder, 'combined_output.vrt')
        gdal.BuildVRT(mosaic_vrt, resampled_vrt_files, separate=True)

        # Create the final geotiff
        product_name = self.s2_metadata['PRODUCT_URI'].replace('.SAFE', '.tiff')
        output_path = os.path.join(self.output_folder, product_name)
        options = ['COMPRESS=LZW', 'TILED=YES']
        gdal.Translate(output_path, mosaic_vrt, format='GTiff', creationOptions=options)

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
        print(f'GeoTIFF saved to: {output_path}')

        # Add path to the metadata
        self.s2_metadata['source_path'] = str(output_path)

        # Clean temp files
        for vrt in resampled_vrt_files:
            gdal.Unlink(vrt)
        gdal.Unlink(mosaic_vrt)

    def run(
        self,
        input_safe: str = None,
        output_folder: str = None,
        roi: str = None,
        target_res: tuple = (20, 20),
        resampling_alg: str = 'near',
        overwrite: bool = False,
    ):
        """
        :param input_safe: Path to the Sentinel-2 SAFE product (can be a .zip file or an unzipped folder).
        :param output_folder: Directory where the resulting geotiff and corresponding metadata will be saved.
        :param roi: Optional - bounding box as a POLYGON wkt string, coordinates must be in WGS84 (EPSG:4326)
        :param target_res: Pixel resolution (x_res, y_res) for the output. Default is 20m.
        :param resampling_alg: The GDAL resampling algorithm (e.g., 'bilinear', 'cubic', 'near'). Default is 'near'.
        :param overwrite: Boolean. If true overwrites any geotiff with same filename present in the output folder.
        """
        self.output_folder = Path(output_folder)
        self.roi = roi
        self.target_res = target_res
        self.resampling_alg = resampling_alg

        # Ensure output folder exists
        self.output_folder.mkdir(parents=True, exist_ok=True)

        product = os.path.basename(input_safe)
        print('Processing ', product)
        path = os.path.join(self.output_folder, product.replace('.zip', '.tiff'))
        if os.path.exists(path) and overwrite is False:
            print('Skipping processing: A geotiff file already exists in the output folder.')
            return

        # check if input safe is a folder or a zip. Unzip to a temporary folder if necessary.
        if os.path.isdir(input_safe):
            self.input_folder = Path(input_safe)
        elif zipfile.is_zipfile(input_safe):
            target_dir = os.path.join(output_folder, 'temp')
            print(f'Unzipping Sentinel-2 SAFE product to: {target_dir}')
            with zipfile.ZipFile(input_safe, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            self.input_folder = os.path.join(
                target_dir, Path(os.path.basename(input_safe).replace('.zip', ''))
            )
        else:
            print(
                'Provided path to the Sentinel-2 SAFE product is neither a folder nor a zip file.'
            )
            return

        # Proceed with the conversion of the SAFE to Geotiff
        print(f'Scanning folder: {self.input_folder}')
        if self._locate_metadata_files():
            self.s2_metadata['Input_SAFE_path'] = str(Path(input_safe))
            self._extract_mtd_msil2a()
            self._extract_mtd_tl()
            self._roi_transform()
            self._process_and_merge_jp2()
            self._save_json()
            if zipfile.is_zipfile(input_safe):
                target_dir = os.path.join(output_folder, 'temp')
                print(f'Removing unzipped Sentinel-2 SAFE product from: {target_dir}')
                shutil.rmtree(target_dir)
        else:
            print('Extraction failed: Required metadata files could not be located.')
