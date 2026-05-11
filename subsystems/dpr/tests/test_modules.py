import json
import os
import zipfile

from pathlib import Path
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import ProjectConfigReader

from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from subsystems.dpr.data_analysis_pipelines import DataAnalysisPipelines
from tests.utils import TestUtils


class TestModules:
    def test_PreprocessingPipelines_001(self):
        """Test PreprocessingPipelines module.

        Check preprocessing pipelines metadata.
        """
        module = PreprocessingPipelines()
        data = module.metadata
        assert isinstance(data, dict) and all(
            isinstance(k, str)
            and isinstance(v, dict)
            and isinstance(v.get('title'), str)
            and isinstance(v.get('abstract'), str)
            and isinstance(v.get('params'), dict)
            for k, v in data.items()
        ), (
            "Invalid structure: expected {str: {'title': str, 'abstract': str, 'params': dict}}"
        )

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
        assert isinstance(data, dict) and all(
            isinstance(k, str)
            and isinstance(v, dict)
            and isinstance(v.get('title'), str)
            and isinstance(v.get('abstract'), str)
            and isinstance(v.get('params'), dict)
            for k, v in data.items()
        ), (
            "Invalid structure: expected {str: {'title': str, 'abstract': str, 'param': dict}}"
        )

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
        data_dir = TestUtils.get_data_path('eou')
        module.set_datasource(data_dir / ('ENMAP01_sample.tif'))
        item_dict = module.stac.create_item()

        with open(data_dir / 'ENMAP01_sample.json', 'r') as f:
            json_dict = json.load(f)
        assert item_dict_no_datetime(item_dict) == item_dict_no_datetime(json_dict)

    def test_MetadataProcessor_002(self, tmp_path):
        """Test MetadataProcessor module.

        Generate data-driven metadata using MetadataGenerator for
        raster-based datasource.
        """

        def item_dict_no_datetime(item_dict):
            if 'properties' in item_dict and 'datetime' in item_dict['properties']:
                del item_dict['properties']['datetime']
            return item_dict

        # first, we need to download the S2 product
        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2026-01-01',
            'end': '2026-01-29',
            'productType': 'S2_MSI_L2A',
        }

        config = ProjectConfigReader(
            Path(__file__).parent.parent.parent.parent
            / 'tests'
            / 'projects'
            / 'jagersfontein.yml'
        )

        dag = DataAcquisitionGateway()

        results = dag.backend.search(
            geom=config['project']['aoi']['geom'],
            **search_filter,
        )

        # S2 product must be extracted for stactools
        s2_path = dag.backend.download(results[0], target_dir=tmp_path)
        with zipfile.ZipFile(s2_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_path)

        # finally, let's generate the metadata
        module = MetadataGenerator()
        module.set_datasource(os.path.splitext(s2_path)[0] + '.SAFE')
        item_dict = module.stac.create_item()

        with open(
            Path(__file__).parent
            / 'sample_data'
            / 'S2B_MSIL2A_20260103T080229_N0511_R035_T35JLH_20260103T103434.json',
            'r',
        ) as f:
            json_dict = json.load(f)
        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)
