from qcl.logger import Logger

class QualityControlLoggingLayer:
    """The Quality Control Layer serves as the critical validation
    gatekeeper within the GAIA-TSF monitoring architecture, situated
    between the ingestion/ETL processes and the Spatial Data
    Infrastructure (SDI) storage."""

    id = "QCL"

    def __init__(self):
        Logger.debug(f"{self.id} initialized")
