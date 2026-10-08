"""Reserved TCN plugin name; implementation is intentionally deferred."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictionResult, PredictiveModel
from subsystems.map.core.registry import register_model


@register_model('tcn')
class TCNModel(PredictiveModel):
    """Future Temporal Convolutional Network contract placeholder."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        raise NotImplementedError(
            'TCNModel is registered but not implemented in Scenario 1.'
        )

    def predict(self, features: np.ndarray) -> PredictionResult:
        raise NotImplementedError(
            'TCNModel is registered but not implemented in Scenario 1.'
        )

    def save(self, path: Path) -> None:
        raise NotImplementedError(
            'TCNModel is registered but not implemented in Scenario 1.'
        )

    @classmethod
    def load(cls, path: Path) -> 'TCNModel':
        raise NotImplementedError(
            'TCNModel is registered but not implemented in Scenario 1.'
        )
