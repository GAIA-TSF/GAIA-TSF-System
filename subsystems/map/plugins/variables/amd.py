from __future__ import annotations

from typing import Any

import numpy as np

from subsystems.map.core.interfaces import VariablePlugin
from subsystems.map.core.registry import register_variable

"""
AMD variable plugin

Encapsulates:
- preprocessing
- feature pipeline selection
- allowed models
"""


@register_variable('amd')
class AMDVariable(VariablePlugin):
    name = 'amd'

    def preprocess(self, data: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """
        AMD index requires strong preprocessing:

        1. Gap filling (clouds, missing acquisitions)
        2. Noise filtering (sensor + atmospheric noise)

        Important:
        This step is CRITICAL for stability of ML models.
        """

        # data = gap_fill(data)
        # data = smooth(data)

        # return data
        return data

    def feature_pipeline(self):
        """
        Could be:
        - temporal
        - lagged

        Currently unified → temporal
        """
        return 'temporal'

    def allowed_models(self):
        """
        Gradient Boosting Regressor:
            - strong for tabular time-series features
        RF:
            - baseline, robust
        """
        return ['constant', 'gbr', 'rf']
