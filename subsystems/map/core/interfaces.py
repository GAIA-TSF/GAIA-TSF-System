"""Target-neutral forecasting-model contract shared by every MAP KV."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class PredictiveModel(ABC):
    """Direct multi-lead forecaster of the change since the forecast origin.

    Every sample is one series (pixel) at one forecast origin ``t``. Pipelines
    pass ``features`` known at ``t`` and ``targets`` of shape ``(samples,
    horizon)`` holding ``y[t + lead] - y_last(t)``, where ``y_last(t)`` is the
    last valid target value at or before ``t``. Missing future observations are
    NaN. ``predict`` returns the same change for leads ``1..horizon``; the
    pipeline adds ``y_last(t)`` back. Models therefore never see units, sign
    conventions, or other KV-specific assumptions.
    """

    def __init__(self, config: dict[str, Any] | None = None, horizon: int = 1) -> None:
        self.config = dict(config or {})
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError('Forecast horizon must be a positive integer.')
        self.horizon = horizon
        self._random_seed = 42

    @abstractmethod
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Fit the model on origin features and ``(samples, horizon)`` changes."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted changes shaped ``(samples, horizon)``."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist this fitted model to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> 'PredictiveModel':
        """Load a fitted model from ``path``."""

    def look_back(self) -> int | None:
        """Number of origin-ending acquisitions a sequence model needs."""
        return None

    def set_random_seed(self, seed: int) -> None:
        """Set the pipeline reproducibility seed."""
        self._random_seed = int(seed)

    def set_validation_data(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Supply held-out data to models that monitor validation loss."""
