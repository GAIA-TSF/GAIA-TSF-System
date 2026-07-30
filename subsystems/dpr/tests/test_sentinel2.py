import json
import os
import shutil
from pathlib import Path
import requests
import tempfile
import numpy
import glob
import pytest
from osgeo import gdal, ogr
from shapely import wkt
from shapely.ops import transform
from pyproj import Transformer

gdal.UseExceptions()

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import Sentinel2SafeProcessor
from subsystems.dpr.data_analysis_pipelines import Sentinel2WaterMaskingPipeline
from tests.utils import TestUtils
from lib.config import SettingsReader

from subsystems.sdi.loader import EarthObservationDataLoader
from subsystems.sdi.utils import SdiUtils


@pytest.fixture(scope='class')
def sentinel2_data():
    """Fixture to download Sentinel-2 data once for all tests in the class."""
    # GDAL configuration to handle errors
    gdal.UseExceptions()

    # Retrieve a Sentinel-2 safe product using EOU's DataAcquisitionGateway
    dag_module = DataAcquisitionGateway()

    # Set ROI as a WKT string (WGS84). This will be used for both product search and cropping parameter.
    roi = 'POLYGON((14.757 60.027, 14.757 60.052, 14.826 60.052, 14.826 60.027, 14.757 60.027))'

    # Set search filters
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2018-07-01',
        'end': '2018-07-31',
        'productType': 'S2_MSI_L2A',
    }

    results = dag_module.backend.search(
        geom=roi,
        **search_filter,
    )
    data_path = dag_module.backend.download(
        results[0], quicklook=False, target_dir='sentinel2'
    )

    # Check if the SAFE product can be located
    assert os.path.exists(data_path), 'The SAFE product was not found.'

    # Return both the data path and ROI for use in tests
    yield {'data_path': data_path, 'roi': roi}


class TestSentinel2Workflow:
    def test_sentinel2_safe_processor(self, sentinel2_data):
        """Test the Sentinel2SafeProcessor."""
        # GDAL configuration to handle errors
        gdal.UseExceptions()

        # Get data from fixture
        data_path = sentinel2_data['data_path']
        roi = sentinel2_data['roi']

        # Define an output folder to store the results
        output_folder = os.path.abspath(
            'subsystems/dpr/tests/sample_data/output_safe_processor'
        )

        # Delete any pre-existing outputs
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Set output resolution
        res = (10, 10)

        # Set resampling algorithm
        r_alg = 'bilinear'

        # Run the pipeline
        pipeline = Sentinel2SafeProcessor()
        pipeline.configure(
            input_safe=Path(data_path),
            output_folder=Path(output_folder),
            roi=roi,
            target_res=res,
            resampling_alg=r_alg,
            overwrite=True,
        )
        pipeline.run()

        # Check if the files were created
        if pipeline._config['output_format'] == 'jp2':
            extension = '.jp2'
        else:
            extension = '.tiff'
        filename = (
            os.path.basename(data_path).replace('.SAFE', '').replace('.zip', '')
            + extension
        )
        raster_path = os.path.join(output_folder, filename)
        assert os.path.exists(raster_path), (
            f'The Geotiff was not created: {raster_path}'
        )
        json_path = os.path.join(output_folder, raster_path.replace(extension, '.json'))
        assert os.path.exists(json_path), 'The metadata file was not created.'

        # Check if essential keys are present in the stac metadata file
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        essential_keys = [
            'type',
            'stac_version',
            'id',
            'properties',
            'geometry',
            'assets',
        ]
        for key in essential_keys:
            assert key in metadata, f'Key {key} is missing from metadata file.'

        # Checks if Geotiff was produced with all the bands and according to inputs
        ds = gdal.Open(raster_path, gdal.GA_Update)
        assert ds.RasterCount == 13, (
            'The Geotiff does not contain the right amount of bands.'
        )

        # Check if output resolution is same as input
        gt = ds.GetGeoTransform()
        output_res = (numpy.abs(gt[1]), numpy.abs(gt[5]))
        assert output_res == res, (
            f'The spatial resolution of the Geotiff {output_res} '
            f'differs from the input parameters {res} .'
        )

        # Check if output ROI is same as input ROI
        # Note: Practically we check if the difference between the bounding boxes coordinates are less than the
        # resolution of 1 pixel (e.g. less or equal to 10m is the input spatial resolution is 10m)
        # Output ROI:
        width = ds.RasterXSize
        height = ds.RasterYSize
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * width
        miny = maxy + gt[5] * height
        output_roi = numpy.array((minx, miny, maxx, maxy))
        # input ROI:
        geom = wkt.loads(roi)
        target_epsg = metadata['properties']['s2:horizontal_cs_code']
        transformer = Transformer.from_crs('EPSG:4326', target_epsg, always_xy=True)
        transformed_geom = transform(transformer.transform, geom)
        input_roi = numpy.array(transformed_geom.bounds)
        # Verify that the offsets are less than the pixel resolution
        offsets = numpy.abs(input_roi - output_roi)
        assert numpy.nanmax(offsets) < numpy.nanmin(res), (
            f'The spatial extent of the Geotiff {output_roi} '
            f'differs by more than {numpy.nanmin(res)} meters from the input parameters {input_roi} .'
        )

        # Check if stac metadata bbox is same as input ROI
        output_bbox = numpy.array(metadata['bbox'])
        xmin, xmax, ymin, ymax = ogr.CreateGeometryFromWkt(roi).GetEnvelope()
        input_bbox = numpy.array((xmin, ymin, xmax, ymax))
        offsets = numpy.abs(input_bbox - output_bbox)
        res_dd = 60.0 / 111320.0  # rough conversion of 60m to dd
        assert numpy.nanmax(offsets) < numpy.nanmin(res_dd), (
            f'The bounding box of the output raster {output_bbox} '
            f'differs by more than {numpy.nanmin(res)} meters from the input bounding box {input_bbox} .'
        )

        # Delete folder after test is complete
        del ds
        shutil.rmtree(output_folder)

    def test_sentinel2_safe_processor_split_bands(self, sentinel2_data):
        """Test the Sentinel2SafeProcessor with split_bands=True."""
        # GDAL configuration to handle errors
        gdal.UseExceptions()

        # Get data from fixture
        data_path = sentinel2_data['data_path']
        roi = sentinel2_data['roi']

        # Define an output folder to store the results
        output_folder = os.path.abspath(
            'subsystems/dpr/tests/sample_data/output_safe_processor_split'
        )

        # Delete any pre-existing outputs
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Set output resolution
        res = (10, 10)

        # Set resampling algorithm
        r_alg = 'bilinear'

        # Run the pipeline with split_bands=True
        pipeline = Sentinel2SafeProcessor()
        pipeline.configure(
            input_safe=Path(data_path),
            output_folder=Path(output_folder),
            roi=roi,
            target_res=res,
            resampling_alg=r_alg,
            split_bands=True,
            overwrite=True,
        )

        # Verify configuration before running
        assert pipeline._config['split_bands'] is True, (
            'split_bands configuration not set correctly'
        )

        pipeline.run()

        # Check if the metadata file was created
        base_filename = (
            os.path.basename(data_path).replace('.SAFE', '').replace('.zip', '')
        )
        json_path = os.path.join(output_folder, base_filename + '.json')
        assert os.path.exists(json_path), 'The metadata file was not created.'

        # Check if essential keys are present in the metadata file
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Debug: print what keys are actually in stac metadata
        print(f'Keys in metadata: {list(metadata.keys())}')

        essential_keys = [
            'type',
            'stac_version',
            'id',
            'properties',
            'geometry',
            'assets',
        ]
        for key in essential_keys:
            assert key in metadata, (
                f'Key {key} is missing from metadata file. Available keys: {list(metadata.keys())}'
            )

        # Check if all 13 band files were created
        if pipeline._config['output_format'] == 'jp2':
            extension = '.jp2'
        else:
            extension = '.tiff'
        expected_bands = [
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
        for band_name in expected_bands:
            band_filename = f'{base_filename}_{band_name}{extension}'
            band_path = os.path.join(output_folder, band_filename)
            assert os.path.exists(band_path), (
                f'Band file {band_filename} was not created.'
            )

        # Check if stac assets were created
        for band_name in expected_bands:
            assert band_name in metadata['assets'], (
                f'Key {band_name} is missing from metadata assets. Available keys: {list(metadata["assets"].keys())}'
            )

        # Check properties of one of the band files (e.g., B02)
        test_band_path = os.path.join(output_folder, f'{base_filename}_B02{extension}')
        ds = gdal.Open(test_band_path, gdal.GA_ReadOnly)

        # Check if it's a single-band file
        assert ds.RasterCount == 1, (
            f'Expected single band in split file, got {ds.RasterCount} bands.'
        )

        # Check if output resolution is same as input
        gt = ds.GetGeoTransform()
        output_res = (numpy.abs(gt[1]), numpy.abs(gt[5]))
        assert output_res == res, (
            f'The spatial resolution of the band file {output_res} '
            f'differs from the input parameters {res}.'
        )

        # Check if output ROI is same as input ROI
        width = ds.RasterXSize
        height = ds.RasterYSize
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * width
        miny = maxy + gt[5] * height
        output_roi = numpy.array((minx, miny, maxx, maxy))

        # input ROI:
        geom = wkt.loads(roi)
        target_epsg = metadata['properties']['s2:horizontal_cs_code']
        transformer = Transformer.from_crs('EPSG:4326', target_epsg, always_xy=True)
        transformed_geom = transform(transformer.transform, geom)
        input_roi = numpy.array(transformed_geom.bounds)

        # Verify that the offsets are less than the pixel resolution
        offsets = numpy.abs(input_roi - output_roi)
        assert numpy.nanmax(offsets) < numpy.nanmin(res), (
            f'The spatial extent of the band file {output_roi} '
            f'differs by more than {numpy.nanmin(res)} meters from the input parameters {input_roi}.'
        )

        # Check if stac metadata bbox is same as input ROI
        output_bbox = numpy.array(metadata['bbox'])
        xmin, xmax, ymin, ymax = ogr.CreateGeometryFromWkt(roi).GetEnvelope()
        input_bbox = numpy.array((xmin, ymin, xmax, ymax))
        offsets = numpy.abs(input_bbox - output_bbox)
        res_dd = 60.0 / 111320.0  # rough conversion of 60m to dd
        assert numpy.nanmax(offsets) < numpy.nanmin(res_dd), (
            f'The bounding box of the output raster {output_bbox} '
            f'differs by more than {numpy.nanmin(res)} meters from the input bounding box {input_bbox} .'
        )

        # Delete folder after test is complete
        del ds
        shutil.rmtree(output_folder)

    def test_sentinel2_safe_processor_export_outputs_to_sdi(self, sentinel2_data):
        """Test the Sentinel2SafeProcessor with zip_output_files=True."""
        # GDAL configuration to handle errors
        gdal.UseExceptions()

        # Get data from fixture
        data_path = sentinel2_data['data_path']
        roi = sentinel2_data['roi']

        # Define an output folder to store the results
        output_folder = os.path.abspath(
            'subsystems/dpr/tests/sample_data/output_safe_processor_zip'
        )

        # Delete any pre-existing outputs
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Set output resolution
        res = (10, 10)

        # Set resampling algorithm
        r_alg = 'bilinear'

        # Run the pipeline with split_bands=True
        pipeline = Sentinel2SafeProcessor()
        pipeline.configure(
            input_safe=Path(data_path),
            output_folder=Path(output_folder),
            roi=roi,
            target_res=res,
            resampling_alg=r_alg,
            split_bands=True,
            overwrite=True,
            zip_output_files=True,
        )

        # Verify configuration before running
        assert pipeline._config['zip_output_files'] is True, (
            'zip_output_files configuration not set correctly'
        )

        pipeline.run()

        # Check if the zip file was created
        base_filename = (
            os.path.basename(data_path).replace('.SAFE', '').replace('.zip', '')
        )
        zip_path = os.path.join(output_folder, base_filename + '.zip')
        assert os.path.exists(zip_path), 'The zip file was not created.'

        utils = SdiUtils()
        # Run the import: uploads raster to S3 and updates STAC
        importer = EarthObservationDataLoader(zip_path=zip_path)
        importer.import_zip()

        # STAC query: search by bbox and datetime
        stac_api_url = importer.stac_api_url
        bbox = importer.stac_json['bbox']
        datetime = importer.stac_json['properties']['datetime']

        query_url = (
            f'{stac_api_url}/search?bbox={",".join(map(str, bbox))}&datetime={datetime}'
        )

        # Send request to STAC API
        resp = requests.post(query_url, json={})
        resp.raise_for_status()
        items = resp.json().get('features', [])
        assert items, 'STAC query returned no items'

        # Find the asset B01
        for stac_item in items:
            if 'B01' in stac_item['assets']:
                asset = stac_item['assets']['B01']
                asset_url = asset['href']

        assert asset_url, 'STAC asset does not contain href'

        # Download the file from STAC asset URL
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        r = requests.get(asset_url, stream=True)
        r.raise_for_status()
        with open(temp_file.name, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        # Compare MD5 hash of downloaded file and input GeoTIFF

        md5_input = utils.file_md5(importer.raster_files[0])
        md5_downloaded = utils.file_md5(temp_file.name)
        assert md5_input == md5_downloaded, (
            'Downloaded file does not match the original GeoTIFF'
        )

    def water_masking_data(self):
        # Set path to folder containing sentinel-2 sample scenes
        input_folder = TestUtils.get_data_path('dpr/sentinel2_yxsjoberg')

        # Check if the folder has been located
        assert input_folder.exists(), 'The Sentinel-2 input folder was not found.'

        # Set path to the vector file containing the water bodies
        input_water_mask = TestUtils.get_data_path('dpr/yxsjoberg_lakes.gpkg')

        # Check if the folder has been located
        assert input_water_mask.exists(), (
            'The vector file containing the water bodies was not found.'
        )

        # Set path to output folder (delete pre-existing folder)
        output_folder = Path(
            SettingsReader()['storage']['data_dir'], 'temp_water_masking_test'
        )
        if output_folder.exists():
            shutil.rmtree(output_folder)

        return input_folder, input_water_mask, output_folder

    def test_sentinel2_water_masking_default(self):
        """Test the Sentinel2 water masking pipeline.
        Run the pipeline in default mode (without user input file, input_water_mask=None)
        """
        input_folder, _, output_folder = self.water_masking_data()

        pipeline = Sentinel2WaterMaskingPipeline()
        pipeline.configure(
            input_folder=input_folder,
            output_folder=output_folder,
            max_cloud_snow_dark=10,
            input_months=None,
            start_date=None,
            end_date=None,
            input_water_mask=None,
            threshold_parameters={
                'shadow index': 0.9,
                'spectral angle': 0.15,
                'vnir regression slope': 1,
                'vnir regression intercept': -500,
                'band 2': 2000,
                'swir reflectance': 1000,
            },
        )
        pipeline.run()

        # Search for metadata and data files in output folder
        input_json_files = glob.glob(os.path.join(input_folder, '*.json'))
        input_data_files = glob.glob(os.path.join(input_folder, '*.jp2'))
        output_json_files = glob.glob(os.path.join(output_folder, '*.json'))
        output_data_files = glob.glob(os.path.join(output_folder, '*.jp2'))

        # Check that all the files were processed and exported correctly
        assert len(input_json_files) == len(output_json_files), (
            f'TEST 1: The number of output metadata files {len(output_json_files)}) is not the same as input metadata '
            f'files ({len(input_json_files)}).'
        )
        assert len(input_data_files) == len(output_data_files), (
            f'TEST 1: The number of output data files ({len(output_data_files)}) is not the same as input data '
            f'files ({len(input_data_files)}).'
        )

        # Check that metadata were updated and that the water mask ratio is below 5%
        for metadata in output_json_files:
            with open(metadata, 'r') as f:
                data = json.load(f)
                assert 'eo:water_mask_percentage' in data['properties'], (
                    f'TEST 1: The key "eo:water_masked_percentage" is missing from metadata file: {metadata}.'
                )
                assert data['properties']['eo:water_mask_percentage'] < 5, (
                    f'TEST 1: The "water_mask_pct" value {data["water_mask_pct"]}% exceed expected threshold (5%).'
                )

        # Check that data files were produced with all the bands and that the mask band (14) contains the expected
        # number of water bodies (16 to 17 for the default workflow).
        for data_file in output_data_files:
            ds = gdal.Open(data_file, gdal.GA_Update)
            assert ds.RasterCount == 14, (
                f'TEST 1: The data file does not contain the right amount of bands ({ds.RasterCount} instead of 14).'
            )
            mask = ds.GetRasterBand(14).ReadAsArray().astype(numpy.int16)
            n_bodies = numpy.nanmax(mask)
            assert 12 <= numpy.nanmax(n_bodies) <= 14, (
                f'TEST 1: Expected between 12 and 14 water bodies, but count is {n_bodies}.'
            )

        # Clear the output folder before running TEST 2
        if output_folder.exists():
            shutil.rmtree(output_folder)

    def test_sentinel2_water_masking_water_input(self):
        """Run the pipeline with vector file as input (input_water_mask)."""
        input_folder, input_water_mask, output_folder = self.water_masking_data()

        pipeline = Sentinel2WaterMaskingPipeline()
        pipeline.configure(
            input_folder=input_folder,
            output_folder=output_folder,
            input_water_mask=input_water_mask,
            max_cloud_snow_dark=10,
            input_months=None,
            start_date=None,
            end_date=None,
            threshold_parameters={
                'shadow index': 0.9,
                'spectral angle': 0.15,
                'vnir regression slope': 1,
                'vnir regression intercept': -500,
                'band 2': 2000,
                'swir reflectance': 1000,
            },
        )
        pipeline.run()

        # Search for metadata and data_file files in output folder
        input_json_files = glob.glob(os.path.join(input_folder, '*.json'))
        input_data_files = glob.glob(os.path.join(input_folder, '*.jp2'))
        output_json_files = glob.glob(os.path.join(output_folder, '*.json'))
        output_data_files = glob.glob(os.path.join(output_folder, '*.jp2'))

        # Check that all the files were processed and exported correctly
        assert len(input_json_files) == len(output_json_files), (
            f'TEST 2: The number of output metadata files {len(output_json_files)}) is not the same as input metadata '
            f'files ({len(input_json_files)}).'
        )
        assert len(input_data_files) == len(output_data_files), (
            f'TEST 2: The number of output data_file files ({len(output_data_files)}) is not the same as input data file '
            f'files ({len(input_data_files)}).'
        )

        # Check that metadata were updated and that the water mask ratio is below 5%
        for metadata in output_json_files:
            with open(metadata, 'r') as f:
                data = json.load(f)
                assert 'eo:water_mask_percentage' in data['properties'], (
                    f'TEST 2: The key "eo:water_masked_percentage" is missing from metadata file: {metadata}.'
                )
                assert data['properties']['eo:water_mask_percentage'] < 5, (
                    f'TEST 2: The "water_mask_pct" value {data["water_mask_pct"]}% exceed expected threshold (5%).'
                )

        # Check that data file files were produced with all the bands and that the mask band (14) contains the expected
        # number of water bodies (10 with the vector file workflow).
        for data_file in output_data_files:
            ds = gdal.Open(data_file, gdal.GA_Update)
            assert ds.RasterCount == 14, (
                f'TEST 2: The data file does not contain the right amount of bands ({ds.RasterCount} instead of 14).'
            )
            mask = ds.GetRasterBand(14).ReadAsArray().astype(numpy.int16)
            n_bodies = numpy.nanmax(mask)
            assert n_bodies == 10, (
                f'TEST 2: Expected 10 water bodies, but count is {n_bodies}.'
            )

        # Clear the output folder
        if output_folder.exists():
            shutil.rmtree(output_folder)
