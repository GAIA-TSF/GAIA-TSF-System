from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.interfaces import PredictiveModel
from core.registry import register_model


@register_model("gbr")
class GBRModel(PredictiveModel):
    """A lightweight gradient-boosting-compatible placeholder for the MAP interface."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Store the training data as a simple baseline model."""
        self.model = {"X": np.asarray(X, dtype=float), "y": np.asarray(y, dtype=float)}

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Generate predictions based on the training mean."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        baseline = float(np.mean(self.model["y"]))
        return np.full(X.shape[0], baseline, dtype=float)

    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "GBRModel":
        """Load a placeholder model."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.config = None
        return instance
