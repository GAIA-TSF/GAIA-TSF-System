from isu import InSituDataUploader


class TestSubsystem:
    def test_ISU_001(self):
        """Test InSituDataUploader subsystem.

        Example of unit test.
        """
        subsystem = InSituDataUploader()
        assert subsystem.id == 'ISU'
