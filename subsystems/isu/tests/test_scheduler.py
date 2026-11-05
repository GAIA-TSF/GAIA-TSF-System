"""Unit tests for the Scheduler module."""

import time
import pytest
from unittest.mock import MagicMock
from lib.scheduler import Scheduler


@pytest.fixture
def mock_qcl_logger():
    """Create a mock logger to prevent NoneType errors after removing default_logger."""
    return MagicMock()


class TestScheduler:
    """
    Unit test suite for the underlying threading Scheduler.
    """

    def test_SCH_001_scheduler_execution(self, mock_qcl_logger):
        """Test if scheduler actually runs the task multiple times."""
        execution_count = {'val': 0}

        def mock_task():
            execution_count['val'] += 1

        sched = Scheduler(interval_seconds=1, logger=mock_qcl_logger)
        sched.start(mock_task)
        time.sleep(2.5)
        sched.stop()

        assert execution_count['val'] >= 2
        assert not sched._is_running

    def test_SCH_002_double_start_prevention(self, mock_qcl_logger):
        """Test that starting twice doesn't crash."""
        sched = Scheduler(interval_seconds=1, logger=mock_qcl_logger)
        sched.start(lambda: None)
        sched.start(lambda: None)

        mock_qcl_logger.warning.assert_called_with('Scheduler is already running.')
        sched.stop()
