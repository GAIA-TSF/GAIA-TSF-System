"""Persistence baseline: repeat the last valid target value for every lead."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from subsystems.map.core.interfaces import PredictiveModel
from subsystems.map.core.registry import register_model


@register_model('constant')
class ConstantBaselineModel(PredictiveModel):
    """Persistence forecast ``y_hat[t + lead] = y_last(t)``.

    Pipelines model the change since the last valid value at the origin, so
    persistence is a zero change for every lead. Nothing is learned; earlier
    versions of this model returned the training-sample mean instead.
    """

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Persistence has no parameters."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return zero change for each sample and lead."""
        return np.zeros((np.asarray(features).shape[0], self.horizon))

    def save(self, path: Path) -> None:
        """Persist the configured horizon with pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> 'ConstantBaselineModel':
        """Load and validate a persistence artifact."""
        with path.open('rb') as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f'Model artifact is not a {cls.__name__}: {path}')
        return model
