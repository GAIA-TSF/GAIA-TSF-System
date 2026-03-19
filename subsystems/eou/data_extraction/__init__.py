from lib.base import GaiaBase, SubsystemId


class DataExtraction(GaiaBase):
    """Data Extraction module acts as the central logic module for
    ingestion. It receives inputs from both the manual loader and the
    acquisition gateway, performing the necessary extraction and
    preparation steps before handing the data off to the downstream
    Data Processing sub-system.
    """

    def __init__(self):
        super().__init__(SubsystemId.EOU)
