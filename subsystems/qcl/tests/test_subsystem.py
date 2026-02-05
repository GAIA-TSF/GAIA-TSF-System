from qcl import QualityControlLoggingLayer


class TestSubsystem:
    def test_QCL_001(self):
        """Test QualityControlLoggingLayer subsystem.
        Example of unit test.
        """
        subsystem = QualityControlLoggingLayer()
        assert subsystem.id == 'QCL'
