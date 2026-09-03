from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Dataset:
    """Container for temporal train/validation/test datasets.

    Arrays are standardized as tabular supervised samples. For raster time
    series, each row represents one pixel at one time step.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    train_time_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    val_time_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    test_time_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    train_pixel_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    val_pixel_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    test_pixel_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    stable_selection_values: np.ndarray | None = None
    raster_shape: tuple[int, ...] | None = None


class VariablePlugin(ABC):
    """
    Encapsulates ALL variable-specific logic.

    Why:
    - Each variable (slope, AMD, etc.) has different preprocessing,
      feature engineering, and allowed models.
    - This prevents scattering logic across the codebase.
    """

    name: str  # unique identifier (used in config)

    @abstractmethod
    def preprocess(self, data, config):
        """
        Perform variable-specific preprocessing.

        Examples:
        - Slope: may do nothing (InSAR already processed)
        - AMD: gap filling + smoothing

        Input:
            raw data (time series)
        Output:
            cleaned/preprocessed data
        """
        pass

    @abstractmethod
    def feature_pipeline(self) -> str:
        """
        Returns name of feature pipeline to use.

        This decouples:
        - WHAT variable is used
        - HOW features are computed

        Example:
            "temporal", "lagged", "temporal_with_gapfill"
        """
        pass

    @abstractmethod
    def allowed_models(self) -> list[str]:
        """
        Restricts which models are valid for this variable.

        Example:
            slope → ["lstm", "rf"]
            amd   → ["xgb", "rf"]

        Prevents invalid combinations.
        """
        pass


class ModelPlugin(ABC):
    """Legacy compatibility interface for model plugins."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def fit(self, X, y):  # noqa: N803
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):  # noqa: N803
        """Return standardized predictions."""
        pass


class PredictiveModel(ABC):
    """Abstract interface for all predictive models."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Train the predictive model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return predictions for the supplied feature matrix."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> "PredictiveModel":
        """Load a trained model from disk."""


class MonitoringPlugin(ABC):
    """Abstract interface for monitoring plugins."""

    name: str = ""

    @abstractmethod
    def evaluate(self, residuals: np.ndarray, config: Any) -> dict[str, Any]:
        """Evaluate residuals and return monitoring output."""


class ExplainabilityPlugin(ABC):
    """Abstract interface for explainability plugins."""

    name: str = ""

    @abstractmethod
    def explain(self, model: Any, X: np.ndarray, config: Any) -> dict[str, Any]:  # noqa: N803
        """Generate explainability artifacts."""


class PredictionResult:
    """Standardized prediction container."""

    def __init__(self, y_pred, uncertainty=None):
        self.y_pred = y_pred
        self.uncertainty = uncertainty
