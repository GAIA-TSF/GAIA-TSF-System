import json
import os
import zipfile

from pathlib import Path

from lib.config import ProjectConfigReader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from subsystems.dpr.data_analysis_pipelines import DataAnalysisPipelines
from tests.utils import TestUtils


def item_dict_no_datetime(item_dict):
    if 'properties' in item_dict and 'datetime' in item_dict['properties']:
        del item_dict['properties']['datetime']
    return item_dict


def get_stac_jsons(product_type: str, search_filter: dict, config, target_dir):
    # first, we need to download the product
    dag = DataAcquisitionGateway()

    results = dag.backend.search(
        geom=config.aoi(),
        productType=product_type,
        **search_filter,
    )

    product_path = dag.backend.download(results[0], target_dir=target_dir)
    product_path_base = os.path.splitext(product_path)[0]
    product_id = os.path.basename(product_path_base)

    # the product must be extracted for stactools
    with zipfile.ZipFile(product_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # finally, let's generate the metadata
    module = MetadataGenerator()
    module.set_datasource(product_path_base + '.SAFE')
    item_dict = module.stac.create_item()

    data_dir = TestUtils.get_data_path('dpr')
    with open(
        Path(data_dir) / f'{product_id}.json',
        'r',
    ) as f:
        json_dict = json.load(f)

    return item_dict, json_dict


class TestModules:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2025-06-01',
        'end': '2025-06-10',
    }
    config = ProjectConfigReader(
        TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    )

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

    def test_MetadataGenerator_001(self):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        raster-based datasource.
        """

        module = MetadataGenerator()
        data_dir = TestUtils.get_data_path('eou')
        module.set_datasource(data_dir / 'ENMAP01_sample.tif')
        item_dict = module.stac.create_item()

        with open(data_dir / 'ENMAP01_sample.json', 'r') as f:
            json_dict = json.load(f)
        assert item_dict_no_datetime(item_dict) == item_dict_no_datetime(json_dict)

    def test_MetadataGenerator_002(self, tmp_path):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        Sentinel-2-based datasource.
        """
        item_dict, json_dict = get_stac_jsons(
            'S2_MSI_L2A',
            search_filter=self.search_filter,
            config=self.config,
            target_dir=tmp_path,
        )

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)

    def test_MetadataGenerator_003(self, tmp_path):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        Sentinel-1-GRD-based datasource.
        """
        item_dict, json_dict = get_stac_jsons(
            'S1_SAR_GRD',
            search_filter=self.search_filter,
            config=self.config,
            target_dir=tmp_path,
        )

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)

    def test_MetadataGenerator_004(self, tmp_path):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        Sentinel-1-SLC-based datasource from CDSE.
        """
        item_dict, json_dict = get_stac_jsons(
            'S1_SAR_SLC',
            search_filter=self.search_filter,
            config=self.config,
            target_dir=tmp_path,
        )

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)

    def test_MetadataGenerator_005(self, tmp_path):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        Sentinel-1-SLC-based datasource from ASF.
        """
        dag = DataAcquisitionGateway(backend='asf')

        search_filter = {
            'direction': 'A',
            'start': '2025-06-01',
            'end': '2025-06-10',
        }

        results = dag.backend.search(
            geom=self.config.aoi(),
            **search_filter,
        )

        products_path = dag.backend.download(results, target_dir=tmp_path)
        product_path = str(products_path / sorted(os.listdir(products_path))[0])
        product_path_base = os.listdir(products_path)[0]

        module = MetadataGenerator()
        module.set_datasource(product_path)
        item_dict = module.stac.create_item()

        product_id = os.path.basename(product_path_base)[:-5]
        data_dir = TestUtils.get_data_path('dpr')
        with open(
            Path(data_dir) / f'{product_id}.json',
            'r',
        ) as f:
            json_dict = json.load(f)

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)
