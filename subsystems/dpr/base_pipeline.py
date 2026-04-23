from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Mapping, Any, Dict

from abc import ABC, abstractmethod

from lib.base import GaiaBase, SubsystemId


class BasePipeline(ABC, GaiaBase):
    metadata = {'title': 'unknown', 'abstract': 'unknown'}

    def __init__(self):
        GaiaBase.__init__(self, SubsystemId.DPR)

    def configure(self, config: Mapping[str, Any]) -> None:
        """Configure pipeline using structured config.

        param Mapping[str, Any]: pipeline configuration in YAML structure
        """
        self._config = config

    @abstractmethod
    def run(self) -> None:
        """Execute pipeline."""
        pass


class PipelineFactory(GaiaBase):
    def __init__(self):
        """Initialize available pipelines."""
        super().__init__(SubsystemId.DPR)
        self._set_pipelines()

    @abstractmethod
    def _set_pipelines(self):
        """Define available pipelines."""
        pass

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
    def pipelines(self):
        return self._pipelines
