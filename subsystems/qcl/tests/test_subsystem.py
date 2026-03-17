from subsystems.qcl import QualityControlLoggingLayer
from lib.base import SubsystemId

class TestSubsystem:
    def test_QCL_001(self):
        """Test QualityControlLoggingLayer subsystem.
        Example of unit test.
        """
        subsystem = QualityControlLoggingLayer()
        assert subsystem.sid == SubsystemId.QCL
