"""
Integration tests for the In-Situ Data Uploader (ISU) Subsystem.
Tests how the Scheduler, Scanner, and Parsing Engine work together.
"""

import os
import shutil
import pytest
import time
from unittest.mock import MagicMock, patch
from subsystems.isu import InSituDataUploader

@pytest.fixture
def mock_subsystem_logger():
    """Mock the global logger used by the subsystem."""
    return MagicMock()

@pytest.fixture
def isu_system(tmp_path, mock_subsystem_logger):
    """
    Initialize the ISU subsystem with temporary directories.
    This ensures we don't mess up the real 'data/input' folder.
    """
    # 1. Create fake input/processed directories
    input_dir = tmp_path / "input"
    processed_dir = tmp_path / "processed"
    input_dir.mkdir()
    processed_dir.mkdir()

    # 2. Initialize the subsystem with these paths
    # We patch the logger to verify logs later
    with patch('subsystems.isu.logger', mock_subsystem_logger):
        isu = InSituDataUploader(
            input_dir=str(input_dir),
            processed_dir=str(processed_dir)
        )
        # Mock the scheduler to prevent actual infinite loops during test
        isu.scheduler = MagicMock()
        return isu

def test_initialization(isu_system):
    """Test if the subsystem initializes its components correctly."""
    assert isu_system.parsing_engine is not None
    assert isu_system.scheduler is not None
    # Verify directories were created (handled by fixture, but good to check logic)
    assert os.path.exists(isu_system.input_dir)
    assert os.path.exists(isu_system.processed_dir)

def test_full_integration_flow(isu_system, tmp_path):
    """
    Critical Integration Test:
    1. Place a file in input_dir.
    2. Trigger the scan method manually.
    3. Verify the file is parsed and moved to processed_dir.
    """
    # 1. Setup: Create a dummy .csv file in the input directory
    input_file = tmp_path / "input" / "sensor_data.csv"
    input_file.write_text("timestamp,value\n2023-01-01,100")

    # 2. Action: Manually trigger the job that the Scheduler would run
    # This proves the "integration" works: Scanner -> Parser -> Archiver
    isu_system._scan_and_process_files()

    # 3. Verification

    # A. Check if ParsingEngine was actually used (by checking logs)
    # The system logger should have recorded success
    # (Note: In a real integration test, we might mock the ParsingEngine to spy on it,
    # but here we rely on the side effect: file movement)

    # B. Check File Lifecycle: File should be GONE from input
    assert not input_file.exists(), "File should be moved from input directory"

    # C. Check File Lifecycle: File should APPEAR in processed
    processed_file = tmp_path / "processed" / "sensor_data.csv"
    assert processed_file.exists(), "File should be archived to processed directory"

def test_scheduler_start_stop(isu_system):
    """Test that start/stop commands trigger the scheduler."""
    isu_system.start()
    isu_system.scheduler.start.assert_called_once()

    isu_system.stop()
    isu_system.scheduler.stop.assert_called_once()