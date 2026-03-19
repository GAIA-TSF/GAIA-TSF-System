from lib.base import GaiaBase, SubsystemId


class SpatialDataInfrastructure(GaiaBase):
    """SpatialDataInfrastructure sub-system"""

    def __init__(self):
        super().__init__(SubsystemId.SDI)
