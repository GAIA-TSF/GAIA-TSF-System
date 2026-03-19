from subsystems.eou import EarthObservationDataUploader
from lib.base import SubsystemId


class TestSubsystem:
    def test_EOU_001(self):
        """Test EarthObservationDataUploader subsystem.

        Example of unit test.
        """
        subsystem = EarthObservationDataUploader()
        assert subsystem.sid == SubsystemId.EOU
