import pytest
import shutil
from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import SettingsReader, ProjectConfigReader
from tests.utils import TestUtils


@pytest.fixture(scope='class')
def project_config():
    return ProjectConfigReader(
        TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    )


class TestModules:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2025-06-01',
        'end': '2025-06-05',
        'productType': 'S2_MSI_L2A',
    }

    def test_ManualFileLoader_001(self):
        """Test ManualFileLoader module.

        Performs file validity test.
        """
        from subsystems.eou.manual_file_loader import ManualFileLoader

        module = ManualFileLoader()
        result = module.check_file_validity(
            TestUtils.get_data_path('eou/ENMAP01_sample.tif')
        )

        assert result['valid'] is True and result['driver'] == 'GTiff'
        assert len(result['errors']) < 1
        assert len(result['warnings']) < 1

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
        assert result[0].product_type == self.search_filter['productType']

    def test_DataAcquisitionGateway_001_asf_search(self, project_config):
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

    def test_DataAcquisitionGateway_002_eodag_download(self, project_config):
        """Test DataAcquisitionGateway module.

        Test download capability using default backend (eodag).
        """
        module = DataAcquisitionGateway()

        results = module.backend.search(
            geom=project_config.aoi(),
            **self.search_filter,
        )

        assert len(results) > 0

        target_dir = 'sentinel2'
        try:
            ql_path = Path(
                module.backend.download(
                    results[0], target_dir=target_dir, quicklook=True
                )
            )

            assert ql_path.exists()
            assert ql_path.stat().st_size > 0
            assert (
                ql_path.parent.resolve()
                == Path(SettingsReader()['storage']['data_dir'], target_dir).resolve()
            )
            visible_files = [
                p for p in ql_path.parent.iterdir() if not p.name.startswith('.')
            ]
            assert len(visible_files) == 1
        finally:
            if ql_path and Path(ql_path).exists():
                Path(ql_path).unlink()

    def test_DataAcquisitionGateway_002_asf_download(self, project_config):
        """Test DataAcquisitionGateway module.

        Test download capability using ASF backend.
        """
        module = DataAcquisitionGateway(backend='asf')
        result = module.backend.search(
            geom=project_config.aoi(),
            start=self.search_filter['start'],
            end=self.search_filter['end'],
            direction='A',
        )

        assert len(result) > 0

        target_dir = 'sentinel1'
        try:
            datadir = Path(
                module.backend.download(result.iloc[[0]], target_dir=target_dir)
            )
            assert any(datadir.iterdir())
            assert (
                datadir.resolve()
                == Path(SettingsReader()['storage']['data_dir'], target_dir).resolve()
            )
            assert len(list(datadir.iterdir())) == 1
        finally:
            if datadir.exists() and datadir.is_dir():
                shutil.rmtree(datadir)

    def test_DataAcquisitionGateway_003_eodag_download_all(self, project_config):
        """Test DataAcquisitionGateway module.

        Test download_all capability using default backend (eodag).
        """
        module = DataAcquisitionGateway()

        results = module.backend.search(
            geom=project_config.aoi(),
            **self.search_filter,
        )

        assert len(results) > 1

        target_dir = 'sentinel2'
        try:
            ql_dir = Path(
                module.backend.download_all(
                    results, target_dir=target_dir, quicklook=True
                )
            )

            assert ql_dir.exists()
            assert any(ql_dir.iterdir())
            assert ql_dir.stat().st_size > 0
            assert (
                ql_dir.resolve()
                == Path(SettingsReader()['storage']['data_dir'], target_dir).resolve()
            )
            visible_files = [p for p in ql_dir.iterdir() if not p.name.startswith('.')]
            assert len(results) == len(visible_files)
        finally:
            if ql_dir and ql_dir.is_dir():
                for item in ql_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

    def test_DataAcquisitionGateway_003_asf_download_all(self, project_config):
        """Test DataAcquisitionGateway module.

        Test download_all capability using ASF backend.
        """
        module = DataAcquisitionGateway(backend='asf')
        result = module.backend.search(
            geom=project_config.aoi(),
            start=self.search_filter['start'],
            end=self.search_filter['end'],
            direction='A',
        )

        assert len(result) > 1

        target_dir = 'sentinel1'
        try:
            datadir = Path(module.backend.download_all(result, target_dir=target_dir))
            assert any(datadir.iterdir())
            assert (
                datadir.resolve()
                == Path(SettingsReader()['storage']['data_dir'], target_dir).resolve()
            )
            assert len(result) == len(list(datadir.iterdir()))
        finally:
            if datadir.exists() and datadir.is_dir():
                shutil.rmtree(datadir)

    def test_DataExtraction_001(self):
        """Test DataExtraction module.

        Example of unit test.
        """
        pass
