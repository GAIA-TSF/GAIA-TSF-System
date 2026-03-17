from lib.base import GaiaBase, SubsystemId

class SpatialDataInfrastructure:
    """SpatialDataInfrastructure sub-system"""

    def __init__(self):
        super().__init__(SubsystemId.SDI)
        self.logger.debug('initialized')
