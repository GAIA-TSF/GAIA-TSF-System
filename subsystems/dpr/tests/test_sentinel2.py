import json
import os
import shutil
from pathlib import Path

import numpy
import glob
import pytest
from osgeo import gdal, ogr
from shapely import wkt
from shapely.ops import transform
from pyproj import Transformer

gdal.UseExceptions()

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import Sentinel2CloudCoverPipeline
from subsystems.dpr.preprocessing_pipelines import Sentinel2SafeProcessor
from subsystems.dpr.data_analysis_pipelines import Sentinel2WaterMaskingPipeline
from tests.utils import TestUtils
from lib.config import SettingsReader


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

        # Check if essential keys are present in the metadata file
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        essential_keys = [
            'PRODUCT_START_TIME',
            'PRODUCT_URI',
            'PROCESSING_LEVEL',
            'PRODUCT_TYPE',
            'DATATAKE_SENSING_START',
            'Wavelengths',
            'Reflectance_Conversion',
            'HORIZONTAL_CS_NAME',
            'HORIZONTAL_CS_CODE',
            'source_paths',
            'Input_SAFE_path',
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
        target_epsg = metadata['HORIZONTAL_CS_CODE']
        transformer = Transformer.from_crs('EPSG:4326', target_epsg, always_xy=True)
        transformed_geom = transform(transformer.transform, geom)
        input_roi = numpy.array(transformed_geom.bounds)
        # Verify that the offsets are less than the pixel resolution
        offsets = numpy.abs(input_roi - output_roi)
        assert numpy.nanmax(offsets) < numpy.nanmin(res), (
            f'The spatial extent of the Geotiff {output_roi} '
            f'differs by more than {numpy.nanmin(res)} meters from the input parameters {input_roi} .'
        )

        # Check if metadata bbox is same as input ROI
        output_bbox = numpy.array(metadata['bbox'])
        input_bbox = numpy.array(ogr.CreateGeometryFromWkt(roi).GetEnvelope())
        offsets = numpy.abs(input_roi - output_roi)
        res_dd = numpy.nanmin(res) / 111320.0  # very rough conversion from meters to dd
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

        # Debug: print what keys are actually in metadata
        print(f'Keys in metadata: {list(metadata.keys())}')

        essential_keys = [
            'PRODUCT_START_TIME',
            'PRODUCT_URI',
            'PROCESSING_LEVEL',
            'PRODUCT_TYPE',
            'DATATAKE_SENSING_START',
            'Wavelengths',
            'Reflectance_Conversion',
            'HORIZONTAL_CS_NAME',
            'HORIZONTAL_CS_CODE',
            'source_paths',
            'Input_SAFE_path',
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

        # Verify that source_path contains all 13 files
        assert 'source_paths' in metadata, 'source_paths key is missing from metadata.'
        assert len(metadata['source_paths']) == 13, (
            f'Expected 13 band files in source_paths, got {len(metadata["source_paths"])}'
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
        target_epsg = metadata['HORIZONTAL_CS_CODE']
        transformer = Transformer.from_crs('EPSG:4326', target_epsg, always_xy=True)
        transformed_geom = transform(transformer.transform, geom)
        input_roi = numpy.array(transformed_geom.bounds)

        # Verify that the offsets are less than the pixel resolution
        offsets = numpy.abs(input_roi - output_roi)
        assert numpy.nanmax(offsets) < numpy.nanmin(res), (
            f'The spatial extent of the band file {output_roi} '
            f'differs by more than {numpy.nanmin(res)} meters from the input parameters {input_roi}.'
        )

        # Check if metadata bbox is same as input ROI
        output_bbox = numpy.array(metadata['bbox'])
        xmin, xmax, ymin, ymax = ogr.CreateGeometryFromWkt(roi).GetEnvelope()
        input_bbox = numpy.array((xmin, ymin, xmax, ymax))
        offsets = numpy.abs(input_bbox - output_bbox)
        res_dd = numpy.nanmin(res) / 111320.0  # very rough conversion from meters to dd
        assert numpy.nanmax(offsets) < numpy.nanmin(res_dd), (
            f'The bounding box of the output raster {output_bbox} '
            f'differs by more than {numpy.nanmin(res)} meters from the input bounding box {input_bbox} .'
        )

        # Delete folder after test is complete
        del ds
        shutil.rmtree(output_folder)

    def test_sentinel2_cloudcover(self):
        """Test the Sentinel2CloudCoverPipeline."""

        # Get path for sentinel-2 sample scene and create a path for test metadata
        raster_path = TestUtils.get_data_path('dpr/sentinel2_clouds.tif')
        metadata_path = raster_path.with_suffix('.json')

        # Check if the raster has been located
        assert os.path.exists(raster_path), 'The Sentinel-2 geotiff was not found.'

        # Check if a metadata file already exists and delete if true
        exists = os.path.exists(metadata_path)
        if exists:
            os.remove(metadata_path)

        # Construct a basic metadata structure
        metadata = {
            'raster_info': {
                'source_paths': [os.path.abspath(raster_path)],
                'filename': os.path.basename(raster_path),
            }
        }

        # Write the metadata to a JSON file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        # Load the metadata to check if it was created
        assert os.path.exists(metadata_path), 'The metadata file was not created.'

        # Run the pipeline
        pipeline = Sentinel2CloudCoverPipeline()
        pipeline.configure(metadata_path=metadata_path, path_key='source_paths')
        pipeline.run()

        # Verification of the outputs
        # get the processed metadata file:
        with open(metadata_path, 'r') as f:
            processed_metadata = json.load(f)

        # Check if the 'cloud_cover_pct' key exists
        assert 'cloud_cover_pct' in processed_metadata, (
            "Key 'cloud_cover_pct' missing from metadata."
        )

        # Check if the value is a valid number between 0 and 1
        cloud_pct = processed_metadata['cloud_cover_pct']
        assert isinstance(cloud_pct, (int, float)), (
            f'Expected number for cloud cover, got {type(cloud_pct)}'
        )
        assert 0 <= cloud_pct <= 1, (
            f'Cloud cover percentage {cloud_pct} is out of bounds (0-1).'
        )

        print(f'Test Passed: Cloud cover derived from SCL band is {cloud_pct}%')

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
            max_cloud_snow_dark=0.1,
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
                assert 'water_mask_pct' in data, (
                    f'TEST 1: The key "water_mask_pct" is missing from metadata file: {metadata}.'
                )
                assert data['water_mask_pct'] < 0.05, (
                    f'TEST 1: The "water_mask_pct" value {data["water_mask_pct"]} exceed expected threshold (0.05).'
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
            max_cloud_snow_dark=0.1,
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
                assert 'water_mask_pct' in data, (
                    f'TEST 2: The key "water_mask_pct" is missing from metadata file: {metadata}.'
                )
                assert data['water_mask_pct'] < 0.05, (
                    f'TEST 2: The "water_mask_pct" value {data["water_mask_pct"]} exceed expected threshold (0.05).'
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
