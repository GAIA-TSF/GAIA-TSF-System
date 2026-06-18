import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from lib.config import ProjectConfigReader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.metadata_processor import MetadataValidator
from subsystems.dpr.metadata_processor.generator import InsituDataset, StacItemFactory
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from subsystems.dpr.data_analysis_pipelines import DataAnalysisPipelines
from tests.utils import TestUtils


_INSITU_META = {
    'ingestion': {'mode': 'manual', 'source': 'sensor.csv'},
    'data': {
        'type': 'in_situ',
        'format': 'csv',
        'sensor_type': 'piezometer',
        'time_range': {
            'start': '2026-01-01T00:00:00Z',
            'end': '2026-06-30T23:59:59Z',
        },
        'schema': ['iso_timestamp', 'lat', 'lon', 'pressure'],
        'location': {'bbox': [14.412, 50.082, 14.440, 50.092]},
        'crs': 'EPSG:4326',
    },
}

_MOCK_COLLECTIONS_RESPONSE = {'collections': [{'id': 'insitu'}]}
_MOCK_COLLECTION_DETAIL = {
    'id': 'insitu',
    'type': 'Collection',
    'stac_version': '1.0.0',
    'description': 'In-situ sensor data',
    'links': [],
    'extent': {
        'spatial': {'bbox': [[-180, -90, 180, 90]]},
        'temporal': {'interval': [[None, None]]},
    },
    'license': 'proprietary',
}


def _make_insitu_stac_json(meta: dict) -> str:
    """Generate an insitu STAC item to a temp JSON file; return path."""
    csv_content = 'iso_timestamp,lat,lon,pressure\n2026-01-01T00:00:00,50.082,14.412,101.3\n'
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False
    ) as csv_f:
        csv_f.write(csv_content)
        csv_path = csv_f.name

    try:
        ds = InsituDataset(csv_path, meta)
        item = StacItemFactory(ds, MagicMock()).create_item()
    finally:
        os.unlink(csv_path)

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as json_f:
        json.dump(item, json_f)
        return json_f.name


def _mock_requests_get(url, *args, **kwargs):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    if url.endswith('/collections'):
        mock.json.return_value = _MOCK_COLLECTIONS_RESPONSE
    else:
        mock.json.return_value = _MOCK_COLLECTION_DETAIL
    return mock


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

    # finally, let's generate the metadata
    module = MetadataGenerator()
    module.set_datasource(product_path)
    item_dict = module.stac.create_item()

    data_dir = TestUtils.get_data_path('dpr')
    with open(
        Path(data_dir) / f'{Path(product_path).with_suffix("").name}.json',
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
        module.set_datasource(TestUtils.get_data_path('eou') / 'ENMAP01_sample.tif')
        item_dict = module.stac.create_item()

        with open(TestUtils.get_data_path('dpr') / 'ENMAP01_sample.json', 'r') as f:
            json_dict = json.load(f)
        assert item_dict_no_datetime(item_dict) == item_dict_no_datetime(json_dict)

    @pytest.mark.slow
    def test_MetadataGenerator_002(self):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        Sentinel-2-based datasource from CDSE.
        """
        item_dict, json_dict = get_stac_jsons(
            'S2_MSI_L2A',
            search_filter=self.search_filter,
            config=self.config,
            target_dir='sentinel2',
        )

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)

    @pytest.mark.slow
    def test_MetadataGenerator_003(self):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        Sentinel-1-GRD-based datasource from CDSE.
        """
        item_dict, json_dict = get_stac_jsons(
            'S1_SAR_GRD',
            search_filter=self.search_filter,
            config=self.config,
            target_dir='sentinel1',
        )

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)

    @pytest.mark.slow
    def test_MetadataGenerator_004(self):
        """Test MetadataGenerator module.

        Generate data-driven metadata using MetadataGenerator for
        Sentinel-1-SLC-based datasource from CDSE.
        """
        item_dict, json_dict = get_stac_jsons(
            'S1_SAR_SLC',
            search_filter=self.search_filter,
            config=self.config,
            target_dir='sentinel1',
        )

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)

    def test_MetadataGenerator_005(self):
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

        product_path = dag.backend.download(results.iloc[[0]], target_dir='sentinel1')

        module = MetadataGenerator()
        module.set_datasource(product_path)
        item_dict = module.stac.create_item()

        data_dir = TestUtils.get_data_path('dpr')
        with open(
            Path(data_dir) / f'{product_path.stem}.json',
            'r',
        ) as f:
            json_dict = json.load(f)

        assert item_dict_no_datetime(
            json.loads(json.dumps(item_dict))
        ) == item_dict_no_datetime(json_dict)

    def test_MetadataValidator_001_eou_valid(self):
        """Test MetadataValidator module with valid metadata."""
        module = MetadataValidator()
        result = module.validate(TestUtils.get_data_path('dpr') / 'ENMAP01_sample.json')

        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert result['valid'] is True
        assert len(result['errors']) < 1

    def test_MetadataValidator_002_eou_invalid_collection(self):
        """Test MetadataValidator with invalid collection reference."""
        module = MetadataValidator()

        # Load original metadata
        original_path = TestUtils.get_data_path('dpr') / 'ENMAP01_sample.json'
        with open(original_path) as f:
            metadata = json.load(f)

        # Modify collection to undefined
        metadata['collection'] = 'undefined'

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(metadata, tmp)
            tmp_path = tmp.name

        try:
            # Validate the modified metadata
            result = module.validate(tmp_path)

            # Assertions
            assert isinstance(result, dict)
            assert result['valid'] is False
            assert len(result['errors']) > 0
            assert any('undefined' in error for error in result['errors'])
        finally:
            # Cleanup
            Path(tmp_path).unlink()

    def test_MetadataValidator_003_eou_missing_properties(self):
        """Test MetadataValidator with missing required properties field."""
        module = MetadataValidator()

        # Load original metadata
        original_path = TestUtils.get_data_path('dpr') / 'ENMAP01_sample.json'
        with open(original_path) as f:
            metadata = json.load(f)

        # Remove properties (required STAC Item field)
        del metadata['bbox']

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(metadata, tmp)
            tmp_path = tmp.name

        try:
            # Validate the modified metadata
            result = module.validate(tmp_path)

            # Assertions - should fail due to missing properties
            assert isinstance(result, dict)
            assert result['valid'] is False
            assert len(result['errors']) > 0
            assert any(
                'properties' in error.lower() or 'datetime' in error.lower()
                for error in result['errors']
            )
        finally:
            # Cleanup
            Path(tmp_path).unlink()

    @patch('subsystems.dpr.metadata_processor.validator.requests.get', side_effect=_mock_requests_get)
    def test_MetadataValidator_001_insitu_valid(self, _mock):
        """Test MetadataValidator with valid in-situ metadata."""
        tmp_path = _make_insitu_stac_json(_INSITU_META)
        try:
            result = MetadataValidator().validate(tmp_path)
            assert isinstance(result, dict)
            assert result['valid'] is True
            assert result['errors'] == []
        finally:
            Path(tmp_path).unlink()

    @patch('subsystems.dpr.metadata_processor.validator.requests.get', side_effect=_mock_requests_get)
    def test_MetadataValidator_002_insitu_missing_time_range(self, _mock):
        """Test MetadataValidator detects missing start_datetime / end_datetime."""
        import copy
        meta = copy.deepcopy(_INSITU_META)
        del meta['data']['time_range']

        tmp_path = _make_insitu_stac_json(meta)
        try:
            result = MetadataValidator().validate(tmp_path)
            assert result['valid'] is False
            assert any('start_datetime' in e for e in result['errors'])
        finally:
            Path(tmp_path).unlink()

    @patch('subsystems.dpr.metadata_processor.validator.requests.get', side_effect=_mock_requests_get)
    def test_MetadataValidator_003_insitu_invalid_bbox(self, _mock):
        """Test MetadataValidator detects bbox outside WGS-84 bounds."""
        import copy
        meta = copy.deepcopy(_INSITU_META)
        meta['data']['location'] = {'bbox': [-200.0, 50.082, 14.440, 50.092]}

        tmp_path = _make_insitu_stac_json(meta)
        try:
            result = MetadataValidator().validate(tmp_path)
            assert result['valid'] is False
            assert any('WGS-84' in e for e in result['errors'])
        finally:
            Path(tmp_path).unlink()

    @patch('subsystems.dpr.metadata_processor.validator.requests.get', side_effect=_mock_requests_get)
    def test_MetadataValidator_004_insitu_start_after_end(self, _mock):
        """Test MetadataValidator detects start_datetime >= end_datetime."""
        import copy
        meta = copy.deepcopy(_INSITU_META)
        meta['data']['time_range'] = {
            'start': '2026-06-30T23:59:59Z',
            'end': '2026-01-01T00:00:00Z',
        }

        tmp_path = _make_insitu_stac_json(meta)
        try:
            result = MetadataValidator().validate(tmp_path)
            assert result['valid'] is False
            assert any('start_datetime' in e for e in result['errors'])
        finally:
            Path(tmp_path).unlink()

    @patch('subsystems.dpr.metadata_processor.validator.requests.get', side_effect=_mock_requests_get)
    def test_MetadataValidator_005_insitu_missing_columns(self, _mock):
        """Test MetadataValidator detects missing table:columns in assets.data."""
        tmp_path = _make_insitu_stac_json(_INSITU_META)
        try:
            with open(tmp_path) as f:
                item_dict = json.load(f)
            item_dict['assets']['data'].pop('table:columns', None)
            with open(tmp_path, 'w') as f:
                json.dump(item_dict, f)

            result = MetadataValidator().validate(tmp_path)
            assert result['valid'] is False
            assert any('table:columns' in e for e in result['errors'])
        finally:
            Path(tmp_path).unlink()

    @patch('subsystems.dpr.metadata_processor.validator.requests.get', side_effect=_mock_requests_get)
    def test_MetadataValidator_006_insitu_invalid_datetime_format(self, _mock):
        """Test MetadataValidator detects invalid datetime format in time range."""
        tmp_path = _make_insitu_stac_json(_INSITU_META)
        try:
            with open(tmp_path) as f:
                item_dict = json.load(f)
            item_dict['properties']['start_datetime'] = 'not-a-datetime'
            with open(tmp_path, 'w') as f:
                json.dump(item_dict, f)

            result = MetadataValidator().validate(tmp_path)
            assert result['valid'] is False
            assert any('datetime' in e.lower() for e in result['errors'])
        finally:
            Path(tmp_path).unlink()
