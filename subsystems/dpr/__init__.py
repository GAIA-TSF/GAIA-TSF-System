from typing import Any, Dict

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

    def process_in_situ(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ISU_I_1: Receive an in-situ CSV file and its metadata from ISU,
        generate a STAC item via MetadataGenerator, and prepare for SDI ingestion.

        :param str file_path: Path to the CSV file provided by ISU.
        :param dict metadata: Metadata dict from ISU (time_range, location, schema, crs, sensor_type).
        :return: Processing result dict with ready_for_sdi flag.
        :rtype: dict
        """
        self.logger.info(f'[DPR] Processing in-situ dataset: {file_path}')
        self.metadata_processor.generator.set_isu_metadata(metadata)
        self.metadata_processor.generator.set_datasource(file_path)
        stac_item = self.metadata_processor.generator.stac.create_item()
        self.logger.info(f'[DPR] STAC item generated for {file_path}')
        return {'status': 'ok', 'ready_for_sdi': True, 'stac_item': stac_item}
