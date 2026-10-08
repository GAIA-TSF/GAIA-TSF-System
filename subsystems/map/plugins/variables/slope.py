from __future__ import annotations

from typing import Any

import numpy as np

from subsystems.map.core.interfaces import VariablePlugin
from subsystems.map.core.registry import register_variable


@register_variable('slope')
class SlopeVariable(VariablePlugin):
    name = 'slope'

    def preprocess(self, data: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """
        InSAR displacement:
        - usually already filtered
        - optionally: detrending, normalization

        Keep minimal to avoid over-processing.
        """
        return data

    def feature_pipeline(self):
        """
        Temporal features:
        - rolling stats
        - velocity / acceleration
        - seasonal indicators
        """
        return 'temporal'

    def allowed_models(self):
        """
        LSTM → captures temporal dynamics
        RF → captures nonlinear patterns in generic engineered features
        tRF → captures nonlinear patterns in DAG-provided temporal features
        """
        return ['constant', 'lstm', 'rf', 'trf']
