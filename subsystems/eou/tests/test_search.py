import pytest

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import ProjectConfigReader
from tests.utils import TestUtils


@pytest.fixture(scope='class')
def project_config():
    return ProjectConfigReader(
        TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    )


class TestEOSearch:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2025-06-01',
        'end': '2025-06-05',
        'productType': 'S2_MSI_L2A',
    }

    def test_DataAcquisitionGateway_001_eodag_search(self, project_config):
        """Test DataAcquisitionGateway module.

        Test search capability using default backend (eodag).
        """
        from eodag.api.search_result import SearchResult

        module = DataAcquisitionGateway()

        result = module.backend.search(
            geom=project_config.aoi(),
            **self.search_filter,
        )

        assert isinstance(result, SearchResult)
        assert len(result) > 0
        assert result[0].collection == self.search_filter['productType']

    def test_DataAcquisitionGateway_002_asf_search(self, project_config):
        """Test DataAcquisitionGateway module.

        Test search capability using ASF backend.
        """
        from geopandas import GeoDataFrame

        module = DataAcquisitionGateway(backend='asf')
        result = module.backend.search(
            geom=project_config.aoi(),
            start=self.search_filter['start'],
            end=self.search_filter['end'],
            direction='A',
        )

        assert isinstance(result, GeoDataFrame)
        assert result is not None
        assert len(result) > 0
