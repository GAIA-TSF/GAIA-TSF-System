from typing import Optional

from lib.base import GaiaBase, SubsystemId

# Import the four core modules
from .etl_engine.pipeline import ETLEngine
from .manual_file_loader import ManualFileLoader
from .bulk_upload_scheduler import BulkUploadScheduler
from .streaming_data_handler import StreamingDataHandler


class InSituDataUploader(GaiaBase):
    """
    GAIA-TSF Prototype: In-Situ Data Uploader (ISU) Subsystem.

    Acts as the **Facade** for the ISU module. It is responsible for initializing
    and managing data ingestion through three concurrent methods: manual upload,
    scheduled background bulk scanning, and real-time streaming. All ingested
    data is routed to the central ETL Engine for unified processing.

    :param project_file: Optional path to a specific project configuration file.
                         Inherited and handled by GaiaBase.
    :type project_file: Optional[str]
    """

    def __init__(self, project_file: Optional[str] = None):
        """
        Initialize the InSituDataUploader subsystem and its sub-components.
        """
        # 1. Initialize the base class and strictly bind the ISU subsystem identity
        super().__init__(SubsystemId.ISU, project_file=project_file)

        self.logger.info('Initializing ISU Subsystem driven by GaiaBase...')

        # 2. Configurations are now handled internally by sub-modules via GaiaBase

        # 3. Initialize the core hub: ETL Engine
        # The engine serves as the central processing unit for all three data streams
        self.etl_engine = ETLEngine(project_file=project_file)

        # 4. Initialize the three parallel ingestion entries
        # The ETL Engine (or its callback) is injected into each loader/handler
        self.manual_loader = ManualFileLoader(
            etl_engine=self.etl_engine, project_file=project_file
        )

        self.bulk_scheduler = BulkUploadScheduler(
            etl_engine=self.etl_engine, project_file=project_file
        )

        self.stream_handler = StreamingDataHandler(
            etl_callback=self.etl_engine.process_data, project_file=project_file
        )

        self.logger.debug('ISU Subsystem components initialized.')

    def start(self) -> None:
        """
        Start all automated data ingestion background tasks.

        This triggers both the scheduled bulk file scanner and the
        real-time streaming listener.

        :return: None
        :rtype: None
        """
        self.logger.info('Starting all ISU background tasks...')
        self.bulk_scheduler.start()
        self.stream_handler.start()

    def stop(self) -> None:
        """
        Gracefully stop all active data ingestion tasks.

        Ensures that schedulers are halted and stream connections are closed properly.

        :return: None
        :rtype: None
        """
        self.logger.info('Stopping all ISU background tasks...')
        self.bulk_scheduler.stop()
        self.stream_handler.stop()
