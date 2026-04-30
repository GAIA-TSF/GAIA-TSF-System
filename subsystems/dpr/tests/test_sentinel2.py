import pytest
import json
import os
import shutil
from pathlib import Path

import numpy
from osgeo import gdal
from shapely import wkt
from shapely.ops import transform
from pyproj import Transformer

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import Sentinel2CloudCoverPipeline
from subsystems.dpr.preprocessing_pipelines import Sentinel2SafeProcessor


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
