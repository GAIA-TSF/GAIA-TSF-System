import json

from pathlib import Path

from dpr.metadata_processor import MetadataGenerator


class TestModules:
    def test_PreprocessingPipelines_001(self):
        """Test PreprocessingPipelines module.

        Example of unit test.
        """
        pass

    def test_DataAnalysisPipelines_001(self):
        """Test DataAnalysisPipelines module.

        Example of unit test.
        """
        pass

    def test_DataAnalysisPipelines_002(self):
        """Test DataAnalysisPipelines module.

        Another example of unit test.
        """
        pass

    def test_MetadataProcessor_001(self):
        """Test MetadataProcessor module.

        Generate data-driven metadata using MetadataGenerator for
        raster-based datasource.
        """

        def item_dict_no_datetime(item_dict):
            if 'properties' in item_dict and 'datetime' in item_dict['properties']:
                del item_dict['properties']['datetime']
            return item_dict

        module = MetadataGenerator(
            Path(__file__).parent.parent.parent
            / 'eou'
            / 'tests'
            / 'sample_data'
            / ('ENMAP01_sample.tif')
        )
        item_dict = module.stac.create_item()

        with open(
            Path(__file__).parent
            / 'sample_data'
            / 'ENMAP01_sample.json',
            'r',
        ) as f:
            json_dict = json.load(f)
        assert item_dict_no_datetime(item_dict) == item_dict_no_datetime(json_dict)
