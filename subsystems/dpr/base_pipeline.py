from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Mapping, Any

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    metadata = {'title': 'unknown', 'abstract': 'unknown'}

    def configure(self, config: Mapping[str, Any]) -> None:
        """Configure pipeline using structured config.

        param Mapping[str, Any]: pipeline configuration in YAML structure
        """
        self._config = config

    @abstractmethod
    def run(self) -> None:
        """Execute pipeline."""
        pass

class PipelineFactory:
    @abstractmethod
    def __init__(self):
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
    def pipeline(self):
        return self._pipelines
