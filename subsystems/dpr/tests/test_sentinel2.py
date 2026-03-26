import json
import os

from subsystems.dpr.preprocessing_pipelines import Sentinel2CloudCoverPipeline


class TestSentinel2Workflow:
    def test_cloudcover(self):
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
