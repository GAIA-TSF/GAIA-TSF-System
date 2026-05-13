from pathlib import Path

from lib.config import ProjectConfigReader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from tests.utils import TestUtils


class TestSentinel1Workflow:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2026-01-01',
        'end': '2026-01-29',
        'productType': 'S1_SAR_SLC',
    }

    def test_download(self):
        """Test EOU Data Acquisition Gateway to download Sentinel-1 data."""
        project_config = ProjectConfigReader(
            TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
        )

        module = DataAcquisitionGateway()
        results = module.backend.search(
            geom=project_config.aoi(),
            **self.search_filter,
        )

        assert len(results) > 0

        ql_path = None
        try:
            ql_path = module.backend.download(
                results[0], target_dir='sentinel1', quicklook=True
            )
            assert Path(ql_path).exists()
        finally:
            if ql_path and Path(ql_path).exists():
                Path(ql_path).unlink()
