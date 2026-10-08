"""Abstract contracts implemented by DAG pipelines, loaders, and extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Pipeline(ABC):
    """Interface for executable DAG pipelines."""

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Run the pipeline.

        Returns:
            Pipeline execution metadata and output locations.
        """


class Plugin(ABC):
    """Base interface for DAG plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""


class RasterLoader(Plugin):
    """Interface for raster time-series loaders."""

    @abstractmethod
    def load(self, directory: Path, filename_pattern: str) -> Any:
        """Load a raster time series from a directory."""


class FeatureExtractor(Plugin):
    """Interface for feature engineering plugins."""

    @abstractmethod
    def compute(
        self,
        data: Any,
        dates: Any,
        enabled_features: dict[str, bool],
    ) -> dict[str, Any]:
        """Compute enabled features from a temporal raster stack."""
