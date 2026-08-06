"""Gradient Boosting Regressor predictive model plugin."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictionResult, PredictiveModel
from subsystems.map.core.registry import register_model


@register_model("gbr")
class GBRModel(PredictiveModel):
    """Sklearn gradient boosting model for future tabular MAP workflows."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ModuleNotFoundError as exc:
            raise RuntimeError("GBRModel requires the optional scikit-learn dependency.") from exc
        self.model = GradientBoostingRegressor(**self.config)

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.model.fit(features, targets)

    def predict(self, features: np.ndarray) -> PredictionResult:
        return PredictionResult(self.model.predict(features))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "GBRModel":
        with path.open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"Model artifact is not a {cls.__name__}: {path}")
        return model
