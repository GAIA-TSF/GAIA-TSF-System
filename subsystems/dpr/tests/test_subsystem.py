from subsystems.dpr import DataProcessing


class TestSubsystem:
    def test_DPR_001(self):
        """Test DataProcessing subsystem.

        Example of unit test.
        """
        subsystem = DataProcessing()
        assert subsystem.id == 'DPR'
