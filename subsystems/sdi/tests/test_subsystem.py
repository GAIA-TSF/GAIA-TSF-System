from sdi import SpatialDataInfrastructure


class TestSubsystem:
    def test_SDI_001(self):
        """Test SpatialDataInfrastructure subsystem.

        Example of unit test.
        """
        subsystem = SpatialDataInfrastructure()
        assert subsystem.id == 'SDI'
