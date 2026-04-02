import json
import os
import shutil
import numpy
from osgeo import gdal

from subsystems.dpr.preprocessing_pipelines import Sentinel2CloudCoverPipeline
from subsystems.dpr.preprocessing_pipelines import Sentinel2SafeProcessor


class TestSentinel2Workflow:
    def manual_test_sentinel2_safe_processor(self):
        """Test the Sentinel2SafeProcessor."""
        # GDAL configuration to handle errors
        gdal.UseExceptions()

        # Get path for sentinel-2 zipped safe product
        safe_path = os.path.abspath(
            'subsystems/dpr/tests/sample_data/S2B_MSIL2A_20180805T102019_N0500_R065_T33VVG_20230731T140510.SAFE.zip'
        )

        # Check if the SAFE product can be located
        assert os.path.exists(safe_path), 'The SAFE product was not found.'

        # Define an output folder to store the results
        output_folder = os.path.abspath(
            'subsystems/dpr/tests/sample_data/output_safe_processor'
        )

        # Delete any pre-existing outputs
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Set ROI
        roi = [486450, 6654430, 490310, 6657230]

        # Set output resolution
        res = (10, 10)

        # Set resampling algorithm
        r_alg = 'bilinear'

        # Run the pipeline
        pipeline = Sentinel2SafeProcessor()
        pipeline.run(
            safe_path, output_folder, roi=roi, target_res=res, resampling_alg=r_alg
        )

        # Check if the files were created
        filename = (
            os.path.basename(safe_path).replace('.SAFE', '').replace('.zip', '.tiff')
        )
        tiff_path = os.path.join(output_folder, filename)
        assert os.path.exists(tiff_path), 'The Geotiff was not created.'
        json_path = os.path.join(output_folder, tiff_path.replace('.tiff', '.json'))
        assert os.path.exists(json_path), 'The metadata file was not created.'

        # Checks if Geotiff was produced with all the bands and according to inputs
        ds = gdal.Open(tiff_path, gdal.GA_Update)
        assert ds.RasterCount == 13, (
            'The Geotiff does not contain the right amount of bands.'
        )
        gt = ds.GetGeoTransform()
        output_res = (numpy.abs(gt[1]), numpy.abs(gt[5]))
        assert output_res == res, (
            f'The spatial resolution of the Geotiff {output_res} '
            f'differs from the input parameters {res} .'
        )
        width = ds.RasterXSize
        height = ds.RasterYSize
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * width
        miny = maxy + gt[5] * height
        output_roi = [minx, miny, maxx, maxy]
        assert output_roi == roi, (
            f'The spatial extent of the Geotiff {output_roi} '
            f'differs from the input parameters {roi} .'
        )

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
        ]
        for key in essential_keys:
            assert key in metadata, f'Key {key} is missing from metadata file.'

    def test_sentinel2_cloudcover(self):
        """Test the Sentinel2CloudCoverPipeline."""

        # Get path for sentinel-2 sample scene and create a path for test metadata
        raster_path = os.path.abspath(
            'subsystems/dpr/tests/sample_data/sentinel2_clouds.tif'
        )
        metadata_path = raster_path.replace('.tif', '.json')

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
        pipeline.run(metadata_path=metadata_path, path_key='source_path', scl_band=13)

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
