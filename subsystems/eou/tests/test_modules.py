from subsystems.eou import EarthObservationDataUploader
from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway

from lib.base import SubsystemId


class TestModules:
    def test_EOU_001(self):
        """Verify the initialization of the EarthObservationDataUploader subsystem.

        Checks that the subsystem is successfully instantiated and assigned
        the expected subsystem identifier.
        """
        subsystem = EarthObservationDataUploader()
        assert subsystem.sid == SubsystemId.EOU

    def test_EOU_002(self):
        """Verify the initialization of the subsystem components.

        Checks that the EarthObservationDataUploader initializes the required
        ManualFileLoader and DataAcquisitionGateway modules and that they are
        instances of the expected classes.
        """
        subsystem = EarthObservationDataUploader()

        assert hasattr(subsystem, "manual_file_loader")
        assert hasattr(subsystem, "data_acquisition_gateway")

        assert isinstance(subsystem.manual_file_loader, ManualFileLoader)
        assert isinstance(subsystem.data_acquisition_gateway, DataAcquisitionGateway)
