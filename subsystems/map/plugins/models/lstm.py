from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.interfaces import PredictiveModel
from core.registry import register_model


@register_model("lstm")
class LSTMModel(PredictiveModel):
    """Placeholder LSTM implementation that is compatible with the new interface."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Train the model. This placeholder keeps the interface consistent for future use."""
        self.model = {"mean_target": float(np.mean(np.asarray(y, dtype=float)))}

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return a simple fallback prediction."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return np.full(X.shape[0], self.model["mean_target"], dtype=float)

    def save(self, path: str | Path) -> None:
        """Persist the model state to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        """Load a placeholder model."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.config = None
        return instance
