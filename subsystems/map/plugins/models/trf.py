"""Temporal Random Forest forecaster over DAG-provided temporal features."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictiveModel
from subsystems.map.core.registry import register_model


@register_model('trf')
class TemporalRandomForestModel(PredictiveModel):
    """Scikit-learn Random Forest, one direct regressor per lead.

    Each row is one pixel at one forecast origin; its columns are the temporal
    features selected in the MAP configuration. Feature engineering remains the
    responsibility of DAG. A separate forest per lead uses every row whose
    target for that lead is observed, so sparse optical gaps at one lead do not
    discard the others.
    """

    def __init__(self, config: dict[str, Any] | None = None, horizon: int = 1) -> None:
        super().__init__(config, horizon)
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ModuleNotFoundError as exc:
            raise RuntimeError('TRF requires the scikit-learn dependency.') from exc
        self.models = [RandomForestRegressor(**self.config) for _ in range(horizon)]

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Fit each lead on rows with an observed target for that lead."""
        for lead, model in enumerate(self.models):
            observed = np.isfinite(targets[:, lead])
            if not np.any(observed):
                raise ValueError(f'No training targets are available for lead {lead + 1}.')
            model.fit(features[observed], targets[observed, lead])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted changes shaped ``(samples, horizon)``."""
        if len(features) == 0:
            return np.empty((0, self.horizon))
        return np.column_stack([model.predict(features) for model in self.models])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> 'TemporalRandomForestModel':
        with path.open('rb') as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f'Model artifact is not a {cls.__name__}: {path}')
        return model
