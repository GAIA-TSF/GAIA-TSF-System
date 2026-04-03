"""
Integration tests for the In-Situ Data Uploader (ISU) Subsystem.
Tests how the Facade, Scheduler, and ETL Engine work together.
"""

import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from subsystems.isu import InSituDataUploader
from lib.base import SubsystemId

TEST_DATA_DIR = Path(__file__).parent / 'test_data'


class TestSubsystem:
    """
    Integration test suite for ISU subsystem.
    """

    @pytest.fixture
    def mock_settings(self):
        """Mock GaiaBase settings to inject temporary directories."""
        return {
            'isu': {
                'input_dir': 'temp_input',
                'processed_dir': 'temp_processed',
                'stream_source_type': 'none',
            }
        }

    @pytest.fixture
    def isu_system(self, tmp_path, mock_settings):
        """
        Initialize the ISU subsystem with temporary directories via mocked GaiaBase.
        """
        input_dir = tmp_path / 'temp_input'
        processed_dir = tmp_path / 'temp_processed'
        input_dir.mkdir()
        processed_dir.mkdir()

        # Patch GaiaBase.settings property to return our mock settings
        with patch('lib.base.GaiaBase.settings', new=mock_settings, create=True):
            isu = InSituDataUploader()

            # Update the paths in the initialized bulk_scheduler to use the exact tmp_path
            isu.bulk_scheduler.input_dir = str(input_dir)
            isu.bulk_scheduler.processed_dir = str(processed_dir)

            # Mock the scheduler thread to prevent infinite loops during test
            isu.bulk_scheduler.scheduler = MagicMock()
            isu.stream_handler.start = MagicMock()
            isu.stream_handler.stop = MagicMock()

            yield isu

    def test_ISU_001_initialization(self, isu_system):
        """Test ISU Subsystem Initialization and Component Wiring."""
        # Verify Identity
        assert getattr(isu_system, 'sid', None) == SubsystemId.ISU

        # Verify Components are wired correctly
        assert isu_system.etl_engine is not None
        assert isu_system.bulk_scheduler is not None
        assert isu_system.manual_loader is not None
        assert isu_system.stream_handler is not None

        # Verify ETL is injected
        assert isu_system.bulk_scheduler.etl_engine == isu_system.etl_engine

    def test_ISU_002_bulk_scan_lifecycle(self, isu_system, tmp_path):
        """
        Test Critical Integration Flow via BulkUploadScheduler:
        File Detection -> ETL Engine Processing -> Archiving.
        """
        # 1. Setup
        source_file = TEST_DATA_DIR / 'slope_sensor_data.csv'
        input_dir = tmp_path / 'temp_input'
        input_file = input_dir / 'slope_sensor_data.csv'

        if not source_file.exists():
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text('dummy,data\n1,2')

        shutil.copy(source_file, input_file)

        # 2. Action
        isu_system.bulk_scheduler._scan_and_process_files()

        # 3. Verification
        # A. Check File Lifecycle: File should be GONE from input
        assert not input_file.exists(), 'File should be moved from input directory'

        # B. Check File Lifecycle: File should APPEAR in processed
        processed_file = tmp_path / 'temp_processed' / 'slope_sensor_data.csv'
        assert processed_file.exists(), 'File should be archived to processed directory'

    def test_ISU_003_start_stop_commands(self, isu_system):
        """Test Facade Start/Stop Commands."""
        # Test Start
        isu_system.start()
        isu_system.bulk_scheduler.scheduler.start.assert_called_once()
        isu_system.stream_handler.start.assert_called_once()

        # Test Stop
        isu_system.stop()
        isu_system.bulk_scheduler.scheduler.stop.assert_called_once()
        isu_system.stream_handler.stop.assert_called_once()
