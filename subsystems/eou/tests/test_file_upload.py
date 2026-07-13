import pytest
import shutil
from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import SettingsReader, ProjectConfigReader
from tests.utils import TestUtils


class TestEOUpload:
    def test_ManualFileLoader_001_check(self):
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


