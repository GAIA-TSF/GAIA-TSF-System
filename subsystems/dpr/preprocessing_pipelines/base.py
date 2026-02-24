from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Mapping, Any

from abc import ABC, abstractmethod

class BasePipeline(ABC):
    metadata = {
        'title': 'unknown',
        'abstract': 'unknown'
    }

    def configure(self, config: Mapping[str, Any]) -> None:
        """Configure pipeline using structured config.

        param Mapping[str, Any]: pipeline configuration in YAML structure
        """
        self._config = config

    @abstractmethod
    def run(self) -> None:
        """Execute pipeline."""
        pass
