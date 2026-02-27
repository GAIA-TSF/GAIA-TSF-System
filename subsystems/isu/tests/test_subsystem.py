"""
Integration tests for the In-Situ Data Uploader (ISU) Subsystem.
Tests how the Scheduler, Scanner, and Parsing Engine work together.
"""

import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from isu import InSituDataUploader

TEST_DATA_DIR = Path(__file__).parent / 'test_data'


class TestSubsystem:
    """
    Integration test suite for ISU subsystem.
    """

    @pytest.fixture
    def mock_logger_class(self):
        """
        Mock the QCL Logger class.
        """
        with patch('subsystems.isu.Logger') as mock_cls:
            # Configure the mock instance that will be returned
            mock_instance = mock_cls.return_value
            yield mock_instance

    @pytest.fixture
    def isu_system(self, tmp_path, mock_logger_class):
        """
        Initialize the ISU subsystem with temporary directories.
        """
        # 1. Create fake input/processed directories to allow safe testing
        input_dir = tmp_path / 'input'
        processed_dir = tmp_path / 'processed'
        input_dir.mkdir()
        processed_dir.mkdir()

        # 2. Initialize the subsystem
        isu = InSituDataUploader(
            input_dir=str(input_dir), processed_dir=str(processed_dir)
        )

        # 3. Mock the scheduler to prevent actual threading/infinite loops during test
        isu.scheduler = MagicMock()

        return isu

    def test_ISU_001(self, isu_system):
        """Test ISU Subsystem Initialization."""
        # Verify Identity
        assert getattr(isu_system, 'id', None) == 'ISU'

        # Verify Components
        assert isu_system.parsing_engine is not None
        assert isu_system.scheduler is not None

        # Verify Directories
        assert os.path.exists(isu_system.input_dir)
        assert os.path.exists(isu_system.processed_dir)

    def test_ISU_002(self, isu_system, tmp_path):
        """
        Test Critical Integration Flow:
        File Detection -> Parsing -> Archiving.
        """
        # 1. Setup
        source_file = TEST_DATA_DIR / 'slope_sensor_data.csv'
        input_dir = tmp_path / 'input'
        input_file = input_dir / 'slope_sensor_data.csv'

        shutil.copy(source_file, input_file)

        # 2. Action: Manually trigger the job that the Scheduler would run
        # This simulates one "tick" of the scheduler
        isu_system._scan_and_process_files()

        # 3. Verification

        # A. Check File Lifecycle: File should be GONE from input
        assert not input_file.exists(), 'File should be moved from input directory'

        # B. Check File Lifecycle: File should APPEAR in processed
        processed_file = tmp_path / 'processed' / 'slope_sensor_data.csv'
        assert processed_file.exists(), 'File should be archived to processed directory'

        # C. Verify Logger was called (proving parsing success)
        # We access the mock logger instance injected into the system
        # Assuming parsing was successful, info logs should happen
        assert isu_system.logger.info.called

    def test_ISU_003(self, isu_system):
        """Test Scheduler Start/Stop Commands."""
        # Test Start
        isu_system.start()
        isu_system.scheduler.start.assert_called_once()

        # Test Stop
        isu_system.stop()
        isu_system.scheduler.stop.assert_called_once()
