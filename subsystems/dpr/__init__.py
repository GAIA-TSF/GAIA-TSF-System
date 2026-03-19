from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from subsystems.dpr.data_analysis_pipelines import DataAnalysisPipelines
from subsystems.dpr.metadata_processor import MetadataProcessor

from lib.base import GaiaBase, SubsystemId


class DataProcessing(GaiaBase):
    """Data Processing sub-system serves as the central refinement
    engine of the architecture, responsible for transforming raw inputs
    into standardized, analysis-ready information products. It encompasses
    three primary functional areas: metadata management, data
    preprocessing, and advanced data analysis. The sub-system ensures that
    all ingested data—whether from satellite imagery or in-situ
    sensors—are accurately described, geometrically and atmospherically
    corrected, and derived into meaningful indicators before storage.
    """

    def __init__(self):
        super().__init__(SubsystemId.DPR)

        self.preprocessing_pipelines = PreprocessingPipelines()
        self.data_analysis_pipelines = DataAnalysisPipelines()
        self.metadata_processor = MetadataProcessor()
