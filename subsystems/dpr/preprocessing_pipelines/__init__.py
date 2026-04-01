from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Any

from .sentinel1 import Sentinel1Pipeline
from .cloudcover import Sentinel2CloudCoverPipeline


class PreprocessingPipelines:
    """The Preprocessing Pipelines are designed as a sequence of
    independent modules tailored for data refinement. Key modules
    include atmospheric correction units, cloud detection services
    (utilizing APIs such as Sen2Cor or Sen2Like), and geometric
    processing tools for georeferencing, orthorectification, and
    reprojection. These pipelines ensure that spatial data is free
    from artifacts and aligned to a common coordinate reference
    system.
    """

    def __init__(self):
        self._pipelines = {
            'sentinel1': Sentinel1Pipeline(),
            'sentinel2_cloudcover': Sentinel2CloudCoverPipeline(),
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get list of registered pipelines.

        :return: list of pipelines (id, metadata)
        :rtype: Dict[str, Any]
        """
        metadata = {}
        for pid, pipeline in self._pipelines.items():
            metadata[pid] = pipeline.metadata

        return metadata

    @property
    def pipeline(self):
        return self._pipelines
