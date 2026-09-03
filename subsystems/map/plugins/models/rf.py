from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.interfaces import PredictiveModel
from core.registry import register_model


@register_model("rf")
class RandomForestModel(PredictiveModel):
    """Random forest regressor with a deterministic NumPy fallback."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Train the predictive model.

        Uses scikit-learn's RandomForestRegressor when available. The fallback
        stores a least-squares model so persistence and inference remain
        operational in minimal environments.
        """
        x_values = np.asarray(X, dtype=float)
        y_values = np.asarray(y, dtype=float)
        random_state = int(getattr(self.config, "random_state", 42))

        try:
            from sklearn.ensemble import RandomForestRegressor

            estimator = RandomForestRegressor(
                n_estimators=int(getattr(self.config, "n_estimators", 200)),
                max_depth=getattr(self.config, "max_depth", None),
                random_state=random_state,
                n_jobs=int(getattr(self.config, "n_jobs", -1)),
            )
            estimator.fit(x_values, y_values)
            self.model = {"kind": "sklearn_random_forest", "estimator": estimator}
            return
        except ImportError:
            pass

        design = np.column_stack([np.ones(x_values.shape[0]), x_values])
        coefficients = np.linalg.pinv(design) @ y_values
        self.model = {"kind": "linear_fallback", "coefficients": coefficients}

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return predictions for the supplied feature matrix."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        x_values = np.asarray(X, dtype=float)
        if self.model["kind"] == "sklearn_random_forest":
            return np.asarray(self.model["estimator"].predict(x_values), dtype=float)

        design = np.column_stack([np.ones(x_values.shape[0]), x_values])
        return np.asarray(design @ self.model["coefficients"], dtype=float)

    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestModel":
        """Load a serialized model from disk."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.config = None
        return instance
