from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway

from subsystems.qcl.logger import Logger


class EarthObservationDataUploader:
    """Earth Observation Data Uploader sub-system is designed to
    manage the acquisition of satellite imagery from both public and
    restricted repositories."""

    id = 'EOU'

    def __init__(self):
        self.logger = Logger(subsystem=self.id)
        self.logger.debug('initialized')

        self.manual_file_loader = ManualFileLoader()
        self.data_acquisition_gateway = DataAcquisitionGateway()
