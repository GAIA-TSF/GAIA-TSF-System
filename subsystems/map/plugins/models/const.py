"""Constant baseline model used as the Scenario 1 reference implementation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictionResult, PredictiveModel
from subsystems.map.core.registry import register_model


@register_model("constant")
class ConstantBaselineModel(PredictiveModel):
    """Learn a single stable-pixel baseline and its residual-scale uncertainty."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.baseline_: float | None = None
        self.uncertainty_: float | None = None

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Estimate the configured central tendency of finite stable targets."""
        del features  # The reference baseline intentionally has no covariates.
        values = np.asarray(targets, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Constant baseline cannot train without finite targets.")
        statistic = str(self.config.get("statistic", "mean"))
        if statistic == "mean":
            self.baseline_ = float(np.mean(values))
        elif statistic == "median":
            self.baseline_ = float(np.median(values))
        else:
            raise ValueError("constant.statistic must be either 'mean' or 'median'.")
        self.uncertainty_ = float(np.std(values))

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Return the learned constant and training standard deviation."""
        if self.baseline_ is None or self.uncertainty_ is None:
            raise RuntimeError("ConstantBaselineModel must be trained before prediction.")
        count = np.asarray(features).shape[0]
        return PredictionResult(
            np.full(count, self.baseline_, dtype=np.float64),
            np.full(count, self.uncertainty_, dtype=np.float64),
        )

    def save(self, path: Path) -> None:
        """Persist the fitted plugin with pickle."""
        if self.baseline_ is None:
            raise RuntimeError("Cannot save an unfitted ConstantBaselineModel.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "ConstantBaselineModel":
        """Load and validate a constant baseline artifact."""
        with path.open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"Model artifact is not a {cls.__name__}: {path}")
        return model
