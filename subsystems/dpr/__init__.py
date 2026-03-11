from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from subsystems.dpr.data_analysis_pipelines import DataAnalysisPipelines
from subsystems.dpr.metadata_processor import MetadataProcessor

from subsystems.qcl.logger import Logger


class DataProcessing:
    """Data Processing sub-system serves as the central refinement
    engine of the architecture, responsible for transforming raw inputs
    into standardized, analysis-ready information products. It encompasses
    three primary functional areas: metadata management, data
    preprocessing, and advanced data analysis. The sub-system ensures that
    all ingested data—whether from satellite imagery or in-situ
    sensors—are accurately described, geometrically and atmospherically
    corrected, and derived into meaningful indicators before storage.
    """

    id = 'DPR'

    def __init__(self):
        self.logger = Logger(subsystem=self.id)
        self.logger.debug('initialized')

        self.preprocessing_pipelines = PreprocessingPipelines()
        self.data_analysis_pipelines = DataAnalysisPipelines()
        self.metadata_processor = MetadataProcessor()
