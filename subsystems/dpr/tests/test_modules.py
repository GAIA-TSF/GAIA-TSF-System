import json

from pathlib import Path

from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from subsystems.dpr.data_analysis_pipelines import DataAnalysisPipelines
from tests.utils import get_data_path


class TestModules:
    def test_PreprocessingPipelines_001(self):
        """Test PreprocessingPipelines module.

        Check preprocessing pipelines metadata.
        """
        module = PreprocessingPipelines()
        data = module.metadata
        assert (
            isinstance(data, dict)
            and all(
                isinstance(k, str)
                and isinstance(v, dict)
                and isinstance(v.get('title'), str)
                and isinstance(v.get('abstract'), str)
                and isinstance(v.get('params'), dict)
                for k, v in data.items()
            )
        ), "Invalid structure: expected {str: {'title': str, 'abstract': str, 'params': dict}}"

        assert len(module.pipelines) > 0
        for name, pipeline in module.pipelines.items():
            assert isinstance(name, str)
            assert isinstance(pipeline.metadata['title'], str)
            assert isinstance(pipeline.metadata['params'], dict)

    def test_DataAnalysisPipelines_001(self):
        """Test PreprocessingPipelines module.

        Check preprocessing pipelines metadata.
        """
        module = DataAnalysisPipelines()
        data = module.metadata
        assert (
            isinstance(data, dict)
            and all(
                isinstance(k, str)
                and isinstance(v, dict)
                and isinstance(v.get('title'), str)
                and isinstance(v.get('abstract'), str)
                and isinstance(v.get('params'), dict)
                for k, v in data.items()
            )
        ), "Invalid structure: expected {str: {'title': str, 'abstract': str, 'param': dict}}"

        assert len(module.pipelines) > 0
        for name, pipeline in module.pipelines.items():
            assert isinstance(name, str)
            assert isinstance(pipeline.metadata['title'], str)
            assert isinstance(pipeline.metadata['params'], dict)

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

        module = MetadataGenerator()
        data_dir = get_data_path('eou')
        module.set_datasource(data_dir / ('ENMAP01_sample.tif'))
        item_dict = module.stac.create_item()

        with open(data_dir / 'ENMAP01_sample.json', 'r') as f:
            json_dict = json.load(f)
        assert item_dict_no_datetime(item_dict) == item_dict_no_datetime(json_dict)
