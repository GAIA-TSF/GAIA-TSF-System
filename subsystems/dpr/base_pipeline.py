from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict

from abc import ABC, abstractmethod

from lib.base import GaiaBase, SubsystemId


class BasePipeline(ABC, GaiaBase):
    metadata = {'title': 'unknown', 'abstract': 'unknown', 'params': None}

    def __init__(self):
        GaiaBase.__init__(self, SubsystemId.DPR)
        self._config = None

    def _configure(self):
        """Set internal properties (to be overridden by pipeline implementation)"""
        pass

    def configure(self, **kwargs) -> None:
        """Configure pipeline using structured config.

        param Mapping[str, Any]: pipeline configuration in YAML structure
        """
        self._configure()
        self._config = kwargs

    def _check_config(self):
        """Check pipeline configuration based on metadata['params']

        Raise RuntimeError on failure
        """
        if (
            self.metadata.get('params', None) is None
            or self.metadata['params'] is None
            or len(self.metadata['params'].keys()) < 1
        ):
            raise RuntimeError('Pipeline has no paramaters defined')

        if self._config is None:
            raise RuntimeError('Pipeline is not configured. Call configure() method')

        for k, v in self.metadata['params'].items():
            if self._config.get(k, None) is None:
                if v.get('default', None) is not None:
                    self._config[k] = v['default']
                else:
                    raise RuntimeError(f"Parameter '{k}' not defined")
            if type(self._config[k]) is not v['dtype']:
                raise RuntimeError(
                    f"Paramater '{k}' (type: {type(self._config[k])}) type mismatch ({v['dtype']})"
                )

    @abstractmethod
    def _run(self) -> None:
        """Execute pipeline."""
        pass

    def run(self) -> None:
        """Execute pipeline."""
        self._check_config()
        return self._run()


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
