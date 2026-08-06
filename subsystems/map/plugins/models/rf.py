"""Random Forest predictive model plugin."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictionResult, PredictiveModel
from subsystems.map.core.registry import register_model


@register_model("rf")
class RFModel(PredictiveModel):
    """Sklearn random-forest model with ensemble-spread uncertainty."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ModuleNotFoundError as exc:
            raise RuntimeError("RFModel requires the optional scikit-learn dependency.") from exc
        self.model = RandomForestRegressor(**self.config)

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.model.fit(features, targets)

    def predict(self, features: np.ndarray) -> PredictionResult:
        predictions = self.model.predict(features)
        per_tree = np.asarray([tree.predict(features) for tree in self.model.estimators_])
        return PredictionResult(predictions, np.std(per_tree, axis=0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "RFModel":
        with path.open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"Model artifact is not a {cls.__name__}: {path}")
        return model
