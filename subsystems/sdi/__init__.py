from qcl.logger import Logger


class SpatialDataInfrastructure:
    """SpatialDataInfrastructure sub-system"""

    id = 'SDI'

    def __init__(self):
        self.logger = Logger(subsystem=self.id)
        self.logger.debug('initialized')
