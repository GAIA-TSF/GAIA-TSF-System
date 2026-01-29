from earth_observation_data_uploader import EarthObservationDataUploader

class TestSubsystem:
    def test_EOU_001(self):
        """Test EarthObservationDataUploader subsystem.

        Example of unit test.
        """
        subsystem = EarthObservationDataUploader()
        assert subsystem.id == "EOU"
