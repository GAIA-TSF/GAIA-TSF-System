from lib.base import GaiaBase, SubsystemId


class ManualFileLoader(GaiaBase):
    """Manual File Loader module for uploading files into the system,
    triggering format detection, metadata checks, schema inspection,
    and version tagging.
    """

    def __init__(self):
        super().__init__(SubsystemId.ISU)
