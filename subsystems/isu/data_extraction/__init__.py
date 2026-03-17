from lib.base import GaiaBase, SubsystemId


class DataExtraction:
    """Data Extraction module performs the core processing logic:
    structural validation, metadata extraction, unit harmonization,
    timestamp normalization, quality checks, and transformation into
    standardized data objects compatible with the GAIA-TSF Spatial
    Data Infrastructure (SDI).
    """

    def __init__(self):
        super().__init__(SubsystemId.ISU)
