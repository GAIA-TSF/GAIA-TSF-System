from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.interfaces import PredictiveModel
from core.registry import register_model


@register_model("xgb")
class XGBoostModel(PredictiveModel):
    """XGBoost-compatible predictive model placeholder.

    The class preserves the MAP model contract without requiring XGBoost as a
    hard dependency. It can be replaced internally with an XGBRegressor without
    changing pipeline code.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model: dict[str, Any] | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Train a deterministic baseline compatible with the predictive API."""
        self.model = {"mean_target": float(np.mean(np.asarray(y, dtype=float)))}

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return predictions for the supplied feature matrix."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return np.full(np.asarray(X).shape[0], self.model["mean_target"], dtype=float)

    def save(self, path: str | Path) -> None:
        """Persist the model state to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        """Load a serialized model from disk."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.config = None
        return instance
