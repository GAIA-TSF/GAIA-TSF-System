"""Unit tests for the Scheduler module."""

import time
import pytest
from unittest.mock import MagicMock
from isu.scheduler import Scheduler


@pytest.fixture
def mock_qcl_logger():
    """Create a mock logger to prevent NoneType errors after removing default_logger."""
    return MagicMock()


class TestScheduler:
    """
    Unit test suite for the Scheduler.
    Follows the OOP pattern to keep tests consistent across the project.
    """

    def test_scheduler_execution(self, mock_qcl_logger):
        """Test if scheduler actually runs the task multiple times."""

        # Counter to track execution using a mutable object
        execution_count = {'val': 0}

        def mock_task():
            execution_count['val'] += 1

        # Initialize scheduler with a very short interval (1 second)
        sched = Scheduler(interval_seconds=1, logger=mock_qcl_logger)

        # Start
        sched.start(mock_task)

        # Wait for 2.5 seconds (should run at least twice: at 0s, 1s, 2s...)
        time.sleep(2.5)

        # Stop
        sched.stop()

        # Assertions
        assert execution_count['val'] >= 2
        assert not sched._is_running

    def test_double_start_prevention(self, mock_qcl_logger):
        """Test that starting twice doesn't crash."""
        sched = Scheduler(interval_seconds=1, logger=mock_qcl_logger)
        sched.start(lambda: None)
        # Should log warning but be safe
        sched.start(lambda: None)
        mock_qcl_logger.warning.assert_called_with('Scheduler is already running.')
        sched.stop()
