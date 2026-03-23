from typing import Any, Optional

from lib.base import GaiaBase, SubsystemId
from subsystems.qcl import QCLayer

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

    External Interfaces:
    - **ISU_I_1 → DPR**: An optional ``dpr_service`` (``DataProcessing`` instance)
      is injected to receive validated and standardised datasets for downstream
      processing. SDI integration is achieved indirectly through this interface
      (ISU_R_07: raw data is converted to SDI-compatible objects before handoff).
    - **ISU_I_2 → QCL**: A ``QCLayer`` instance is created internally and injected
      into the ETL Engine and StreamingDataHandler. ISU reports ingestion status,
      validation results, and error metadata to QCL as structured events.

    Note: ISU has **no direct interface with SDI**. SDI integration is indirect,
    via the data products and metadata delivered to DPR through ISU_I_1.

    :param project_file: Optional path to a specific project configuration file.
    :type project_file: Optional[str]
    :param dpr_service: Optional Data Processing subsystem instance (ISU_I_1).
    :type dpr_service: Any
    """

    def __init__(
        self,
        project_file: Optional[str] = None,
        dpr_service: Any = None,
    ):
        """
        Initialize the InSituDataUploader subsystem and its sub-components.
        """
        # 1. Initialize the base class and strictly bind the ISU subsystem identity
        super().__init__(SubsystemId.ISU, project_file=project_file)

        self.logger.info("Initializing ISU Subsystem driven by GaiaBase...")

        # 2. ISU_I_2: Wire QCL — instantiate and own the Quality Control layer.
        # QCL receives ingestion status, validation results, and error reports.
        self.qc_layer = QCLayer()
        self.logger.debug("QCLayer (ISU_I_2) wired into ISU.")

        # 3. ISU_I_1: Store the optional DPR service reference.
        # SDI integration is indirect: ISU converts raw data to SDI-compatible
        # objects (ISU_R_07) and delivers them to DPR, which handles SDI ingestion.
        self.dpr_service = dpr_service

        # 4. Initialize the core hub: ETL Engine with all service dependencies
        self.etl_engine = ETLEngine(
            project_file=project_file,
            qc_layer=self.qc_layer,
            dpr_service=self.dpr_service,
        )

        # 5. Initialize the three parallel ingestion entries
        self.manual_loader = ManualFileLoader(
            etl_engine=self.etl_engine,
            project_file=project_file
        )

        self.bulk_scheduler = BulkUploadScheduler(
            etl_engine=self.etl_engine,
            project_file=project_file
        )

        self.stream_handler = StreamingDataHandler(
            etl_callback=self.etl_engine.process_data,
            qc_layer=self.qc_layer,
            project_file=project_file
        )

        self.logger.debug("ISU Subsystem components initialized.")

    def start(self) -> None:
        """
        Start all automated data ingestion background tasks.
        
        This triggers both the scheduled bulk file scanner and the 
        real-time streaming listener.
        
        :return: None
        :rtype: None
        """
        self.logger.info("Starting all ISU background tasks...")
        self.bulk_scheduler.start()
        self.stream_handler.start()

    def stop(self) -> None:
        """
        Gracefully stop all active data ingestion tasks.
        
        Ensures that schedulers are halted and stream connections are closed properly.

        :return: None
        :rtype: None
        """
        self.logger.info("Stopping all ISU background tasks...")
        self.bulk_scheduler.stop()
        self.stream_handler.stop()