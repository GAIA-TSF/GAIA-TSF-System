"""Unit tests for the Scheduler module."""

import pytest
import time
from subsystems.isu.scheduler import Scheduler


def test_scheduler_execution():
    """Test if scheduler actually runs the task multiple times."""

    # Counter to track execution using a mutable object
    execution_count = {'val': 0}

    def mock_task():
        execution_count['val'] += 1

    # Initialize scheduler with a very short interval (1 second)
    sched = Scheduler(interval_seconds=1)

    # Start
    sched.start(mock_task)

    # Wait for 2.5 seconds (should run at least twice: at 0s, 1s, 2s...)
    time.sleep(2.5)

    # Stop
    sched.stop()

    # Assertions
    assert execution_count['val'] >= 2
    assert not sched._is_running


def test_double_start_prevention():
    """Test that starting twice doesn't crash."""
    sched = Scheduler(interval_seconds=1)
    sched.start(lambda: None)
    sched.start(lambda: None)  # Should log warning but be safe
    sched.stop()