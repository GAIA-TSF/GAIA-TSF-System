from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.eou.data_extraction import DataExtraction

from subsystems.qcl.logger import Logger
from lib.base import BaseObject, SubsystemId


class EarthObservationDataUploader(BaseObject):
    """Earth Observation Data Uploader sub-system is designed to
    manage the acquisition of satellite imagery from both public and
    restricted repositories."""

    def __init__(self):
        super().__init__(SubsystemId.EOU)

        self.manual_file_loader = ManualFileLoader()
        self.data_acquisition_gateway = DataAcquisitionGateway()
        self.data_extraction = DataExtraction()
