from isu.manual_file_loader import ManualFileLoader
from isu.bulk_upload_scheduler import BulkUploadScheduler
from isu.streaming_data_handler import StreamingDataHandler
from isu.data_extraction import DataExtraction

from qcl.logger import Logger


class InSituDataUploader:
    """In-Situ Data Uploader sub-system is responsible for collecting
    and securely transmitting field-acquired data from different
    ground-based sensor technologies to the central data
    pre-processing module.
    """

    id = 'ISU'

    def __init__(self):
        self.logger = Logger(subsystem=self.id)
        self.logger.debug('initialized')

        self.manual_file_loader = ManualFileLoader()
        self.bulk_upload_scheduler = BulkUploadScheduler()
        self.streaming_data_handler = StreamingDataHandler()
        self.data_extraction = DataExtraction()
