"""Reserved LSTM plugin name for a future sequence-model implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictionResult, PredictiveModel
from subsystems.map.core.registry import register_model


@register_model("lstm")
class LSTMModel(PredictiveModel):
    """Future LSTM contract placeholder, deliberately unavailable in Scenario 1."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        raise NotImplementedError("LSTMModel is not implemented in Scenario 1.")

    def predict(self, features: np.ndarray) -> PredictionResult:
        raise NotImplementedError("LSTMModel is not implemented in Scenario 1.")

    def save(self, path: Path) -> None:
        raise NotImplementedError("LSTMModel is not implemented in Scenario 1.")

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        raise NotImplementedError("LSTMModel is not implemented in Scenario 1.")
