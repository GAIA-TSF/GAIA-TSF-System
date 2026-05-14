from subsystems.dpr.base_pipeline import PipelineFactory
from .sentinel1 import Sentinel1Pipeline
from .cloudcover import Sentinel2CloudCoverPipeline
from .sentinel2safe import Sentinel2SafeProcessor


class PreprocessingPipelines(PipelineFactory):
    """The Preprocessing Pipelines are designed as a sequence of
    independent modules tailored for data refinement. Key modules
    include atmospheric correction units, cloud detection services
    (utilizing APIs such as Sen2Cor or Sen2Like), and geometric
    processing tools for georeferencing, orthorectification, and
    reprojection. These pipelines ensure that spatial data is free
    from artifacts and aligned to a common coordinate reference
    system.
    """

    def _set_pipelines(self):
        """Define available preprocessing pipelines."""
        self._pipelines = {
            'sentinel1': Sentinel1Pipeline(),
            'sentinel2_safe_processor': Sentinel2SafeProcessor(),
            'sentinel2_cloudcover': Sentinel2CloudCoverPipeline(),
        }
