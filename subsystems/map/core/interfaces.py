"""Stable public abstractions for MAP plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class VariablePlugin(ABC):
    """Encapsulate preprocessing and model compatibility for one variable."""

    name: str

    @abstractmethod
    def preprocess(self, data: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """Return variable-specific preprocessed data."""

    @abstractmethod
    def feature_pipeline(self) -> str:
        """Return the default feature pipeline name."""

    @abstractmethod
    def allowed_models(self) -> list[str]:
        """Return model plugin names allowed for this variable."""


class PredictionResult:
    """Prediction values and optional per-sample uncertainty."""

    def __init__(
        self,
        y_pred: np.ndarray,
        uncertainty: np.ndarray | None = None,
    ) -> None:
        self.y_pred = np.asarray(y_pred, dtype=np.float64)
        self.uncertainty = (
            None if uncertainty is None else np.asarray(uncertainty, dtype=np.float64)
        )


class PredictiveModel(ABC):
    """Common persistence and inference contract for predictive model plugins."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Fit the model using feature rows and observed targets."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionResult:
        """Predict expected observations for feature rows."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist this fitted model to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "PredictiveModel":
        """Load a fitted model from ``path``."""

    # Backward compatible aliases for early MAP plugins.
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Alias for :meth:`train`."""
        self.train(features, targets)


ModelPlugin = PredictiveModel
