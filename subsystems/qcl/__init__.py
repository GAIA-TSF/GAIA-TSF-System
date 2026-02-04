from qcl.logger import Logger


class QualityControlLoggingLayer:
    """The Quality Control Layer serves as the critical validation
    gatekeeper within the GAIA-TSF monitoring architecture, situated
    between the ingestion/ETL processes and the Spatial Data
    Infrastructure (SDI) storage."""

    id = 'QCL'

    def __init__(self):
        self.logger = Logger(subsystem=self.id)
        self.logger.debug(f'{self.id} initialized')
