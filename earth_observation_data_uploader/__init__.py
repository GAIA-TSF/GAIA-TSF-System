from earth_observation_data_uploader.manual_file_loader import ManualFileLoader
from earth_observation_data_uploader.data_acquisition_gateway import DataAcquisitionGateway
from earth_observation_data_uploader.data_extraction import DataExtraction

# comes from Quality-Control-and-Logging-Layer
# from qcl.logger import Logger

class EarthObservationDataUploader:
    """Earth Observation Data Uploader sub-system is designed to
    manage the acquisition of satellite imagery from both public and
    restricted repositories."""

    id = "EOU"

    def __init__(self):
        # Logger.debug(f"{self.id} initialized")

        self.manual_file_loader = ManualFileLoader()
        self.data_acquisition_gateway = DataAcquisitionGateway()
        self.data_extraction = DataExtraction()
