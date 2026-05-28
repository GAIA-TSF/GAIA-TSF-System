import json
import os
import shutil
from pathlib import Path

import numpy
import glob
from osgeo import gdal
from shapely import wkt
from shapely.ops import transform
from pyproj import Transformer

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import Sentinel2CloudCoverPipeline
from subsystems.dpr.preprocessing_pipelines import Sentinel2SafeProcessor
from subsystems.dpr.data_analysis_pipelines import Sentinel2WaterMaskingPipeline
from tests.utils import TestUtils


class TestSentinel2Workflow:
    def test_sentinel2_safe_processor(self):
        """Test the Sentinel2SafeProcessor."""
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
        filename = (
            os.path.basename(data_path).replace('.SAFE', '').replace('.zip', '')
            + '.tiff'
        )
        tiff_path = os.path.join(output_folder, filename)
        assert os.path.exists(tiff_path), f'The Geotiff was not created: {tiff_path}'
        json_path = os.path.join(output_folder, tiff_path.replace('.tiff', '.json'))
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
            'source_path',
            'Input_SAFE_path',
        ]
        for key in essential_keys:
            assert key in metadata, f'Key {key} is missing from metadata file.'

        # Checks if Geotiff was produced with all the bands and according to inputs
        ds = gdal.Open(tiff_path, gdal.GA_Update)
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

        # Delete folder after test is complete
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
                'source_path': os.path.abspath(raster_path),
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
        pipeline.configure(metadata_path=metadata_path, path_key='source_path')
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

    def test_sentinel2_water_masking(self):
        """Test the Sentinel2 water masking pipeline."""
        # GDAL configuration to handle errors
        gdal.UseExceptions()

        # Set path to folder containing sentinel-2 sample scenes
        input_folder = os.path.abspath('tests/data/dpr/sentinel2_yxsjoberg')

        # Check if the folder has been located
        assert os.path.exists(input_folder), (
            'The Sentinel-2 input folder was not found.'
        )

        # Set path to the vector file containing the water bodies
        input_water_mask = os.path.abspath('tests/data/dpr/yxsjoberg_lakes.gpkg')

        # Check if the folder has been located
        assert os.path.exists(input_water_mask), (
            'The vector file containing the water bodies was not found.'
        )

        # Set path to output folder (delete pre-existing folder)
        output_folder = os.path.abspath('tests/data/dpr/temp_water_masking_test')
        if os.path.exists(output_folder) and os.path.isdir(output_folder):
            shutil.rmtree(output_folder)

        # TEST 1: Run the pipeline in default mode (without user input file, input_water_mask=None)
        pipeline = Sentinel2WaterMaskingPipeline()
        pipeline.configure(
            input_folder=Path(input_folder),
            output_folder=Path(output_folder),
            max_cloud_snow_dark=0.1,
            input_months=[4, 5, 6, 7, 8, 9, 10],
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

        # Search for metadata and geotiff files in output folder
        input_json_files = glob.glob(os.path.join(input_folder, '*.json'))
        input_tiff_files = glob.glob(os.path.join(input_folder, '*.tif*'))
        output_json_files = glob.glob(os.path.join(output_folder, '*.json'))
        output_tiff_files = glob.glob(os.path.join(output_folder, '*.tif*'))

        # Check that all the files were processed and exported correctly
        assert len(input_json_files) == len(output_json_files), (
            f'TEST 1: The number of output metadata files {len(output_json_files)}) is not the same as input metadata '
            f'files ({len(input_json_files)}).'
        )
        assert len(input_tiff_files) == len(output_tiff_files), (
            f'TEST 1: The number of output geotiff files ({len(output_tiff_files)}) is not the same as input geotiff '
            f'files ({len(input_tiff_files)}).'
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

        # Check that Geotiff files were produced with all the bands and that the mask band (14) contains the expected
        # number of water bodies (16 to 17 for the default workflow).
        for geotiff in output_tiff_files:
            ds = gdal.Open(geotiff, gdal.GA_Update)
            assert ds.RasterCount == 14, (
                f'TEST 1: The Geotiff does not contain the right amount of bands ({ds.RasterCount} instead of 14).'
            )
            mask = ds.GetRasterBand(14).ReadAsArray().astype(numpy.int16)
            n_bodies = numpy.nanmax(mask)
            assert 12 <= numpy.nanmax(n_bodies) <= 14, (
                f'TEST 1: Expected between 12 and 14 water bodies, but count is {n_bodies}.'
            )

        # Clear the output folder before running TEST 2
        if os.path.exists(output_folder) and os.path.isdir(output_folder):
            shutil.rmtree(output_folder)

        # TEST 2: Run the pipeline with vector file as input (input_water_mask)
        pipeline = Sentinel2WaterMaskingPipeline()
        pipeline.configure(
            input_folder=Path(input_folder),
            output_folder=Path(output_folder),
            input_water_mask=Path(input_water_mask),
            max_cloud_snow_dark=0.1,
            input_months=[4, 5, 6, 7, 8, 9, 10],
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

        # Search for metadata and geotiff files in output folder
        input_json_files = glob.glob(os.path.join(input_folder, '*.json'))
        input_tiff_files = glob.glob(os.path.join(input_folder, '*.tif*'))
        output_json_files = glob.glob(os.path.join(output_folder, '*.json'))
        output_tiff_files = glob.glob(os.path.join(output_folder, '*.tif*'))

        # Check that all the files were processed and exported correctly
        assert len(input_json_files) == len(output_json_files), (
            f'TEST 2: The number of output metadata files {len(output_json_files)}) is not the same as input metadata '
            f'files ({len(input_json_files)}).'
        )
        assert len(input_tiff_files) == len(output_tiff_files), (
            f'TEST 2: The number of output geotiff files ({len(output_tiff_files)}) is not the same as input geotiff '
            f'files ({len(input_tiff_files)}).'
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

        # Check that Geotiff files were produced with all the bands and that the mask band (14) contains the expected
        # number of water bodies (10 with the vector file workflow).
        for geotiff in output_tiff_files:
            ds = gdal.Open(geotiff, gdal.GA_Update)
            assert ds.RasterCount == 14, (
                f'TEST 2: The Geotiff does not contain the right amount of bands ({ds.RasterCount} instead of 14).'
            )
            mask = ds.GetRasterBand(14).ReadAsArray().astype(numpy.int16)
            n_bodies = numpy.nanmax(mask)
            assert n_bodies == 10, (
                f'TEST 2: Expected 10 water bodies, but count is {n_bodies}.'
            )

        # Clear the output folder
        if os.path.exists(output_folder) and os.path.isdir(output_folder):
            shutil.rmtree(output_folder)
