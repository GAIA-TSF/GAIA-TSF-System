import pytest

from subsystems.dpr import DataProcessing


class TestSubsystem:
    @pytest.mark.skip(reason='Broken until #165 is solved')
    def test_DPR_001(self):
        """Test DataProcessing subsystem.

        Example of unit test.
        """
        subsystem = DataProcessing()
        assert subsystem.id == 'DPR'
