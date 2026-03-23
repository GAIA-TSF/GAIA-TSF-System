import pytest

from subsystems.dpr import DataProcessing
from lib.base import SubsystemId

class TestSubsystem:
    def test_DPR_001(self):
        """Test DataProcessing subsystem.

        Example of unit test.
        """
        subsystem = DataProcessing()
        assert subsystem.sid == SubsystemId.DPR
