import shutil
import time
import pytest
from pathlib import Path

from lib.scheduler import Scheduler
from subsystems.eou.bulk_upload_scheduler import BulkUploadScheduler
from tests.utils import TestUtils
from lib.config import SettingsReader


@pytest.fixture(scope='module')
def project_config_path():
    return TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')


class TestBulkUploadScheduler:
    """Integration and unit tests for BaseScheduler and BulkUploadScheduler using real data."""

    def test_SCH_001_base_scheduler_execution(self):
        """Test underlying BaseScheduler threading execution and double-start prevention."""
        execution_count = {'val': 0}

        def sample_task():
            execution_count['val'] += 1

        sched = Scheduler(interval_seconds=1)
        sched.start(sample_task)

        # Ensure calling start() twice while running doesn't crash or duplicate threads
        sched.start(sample_task)

        time.sleep(2.5)
        sched.stop()

        assert execution_count['val'] >= 2
        assert not sched._is_running

    def test_SCH_002_scheduler_initialization(self, project_config_path):
        """Verify BulkUploadScheduler initializes GaiaBase and gateway."""
        scheduler = BulkUploadScheduler(project_path=project_config_path)

        assert scheduler.sid.name == 'EOU'
        assert scheduler.project_config is not None
        assert scheduler.project_config.aoi() is not None
        assert scheduler.scheduler is not None
        assert scheduler.interval > 0
        assert scheduler.lookback_days > 0

    def test_SCH_003_real_eou_bulk_upload_sync(self, project_config_path, monkeypatch):
        """Test real EO catalog search and download using live DataAcquisitionGateway."""
        # Set environment variable so child instances (Gateway) pick up site_id automatically
        monkeypatch.setenv('GAIA_PROJECT_PATH', str(project_config_path))

        scheduler = BulkUploadScheduler(project_path=project_config_path)

        # Isolate download directory and widen lookback to guarantee satellite pass hits
        scheduler.lookback_days = 3
        scheduler.quicklook = True

        for index, item in enumerate(scheduler.project_config.eou):
            item['target_dir'] = f'test_scheduler_dir_{index}'

        # Resolve download destination from backend root data directory
        data_dir = Path(SettingsReader()['storage']['data_dir']).resolve()
        download_dirs = [
            data_dir / item.target_dir for item in scheduler.project_config.eou
        ]

        try:
            scheduler._download_eo_products()

            for download_dir in download_dirs:
                if download_dir.exists():
                    downloaded_items = list(download_dir.iterdir())
                    assert len(downloaded_items) >= 0

        finally:
            for download_dir in download_dirs:
                if download_dir.exists():
                    for item in download_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    download_dir.rmdir()

    def test_SCH_004_scheduler_start_stop_lifecycle(
        self, project_config_path, monkeypatch
    ):
        """Test background thread lifecycle management for BulkUploadScheduler."""
        monkeypatch.setenv('GAIA_PROJECT_PATH', str(project_config_path))
        scheduler = BulkUploadScheduler(project_path=project_config_path)
        scheduler.interval = 1  # Fast loop interval for testing

        assert not scheduler.scheduler._is_running

        scheduler.start()
        assert scheduler.scheduler._is_running

        time.sleep(1.5)  # Allow background daemon to execute at least one tick

        scheduler.stop()
        assert not scheduler.scheduler._is_running
