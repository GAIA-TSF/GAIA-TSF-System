from subsystems.dpr.base_pipeline import PipelineFactory

from .water_masking import Sentinel2WaterMaskingPipeline


class DataAnalysisPipelines(PipelineFactory):
    """The Data Analysis and Derivation Pipelines represent the
    higher-level processing layer. These modules operate on the
    preprocessed data to compute spectral indices (e.g., NDVI, NDWI)
    and execute classification workflows for land cover or
    mineralogical mapping. Each analysis module is configurable,
    allowing the system to adapt to specific project objectives or
    data types.
    """

    def _set_pipelines(self):
        """Define available data analysis pipelines."""
        self._pipelines = {'sentinel2_water_masking': Sentinel2WaterMaskingPipeline()}
