from lib.base import GaiaBase, SubsystemId


class QualityControlLoggingLayer(GaiaBase):
    """The Quality Control Layer serves as the critical validation
    gatekeeper within the GAIA-TSF monitoring architecture, situated
    between the ingestion/ETL processes and the Spatial Data
    Infrastructure (SDI) storage."""
    def __init__(self):
        super().__init__(SubsystemId.QCL)

