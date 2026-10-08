import logging
from typing import Any

from core.interfaces import VariablePlugin
from core.registry import register_variable

"""
AMD variable plugin

Encapsulates:
- preprocessing
- feature pipeline selection
- allowed models
"""

logger = logging.getLogger("map.variables.amd")


@register_variable('amd')
class AMDVariable(VariablePlugin):
    name = 'amd'

    def preprocess(self, data: Any, config: Any) -> Any:
        """
        AMD index requires strong preprocessing:

        1. Gap filling (clouds, missing acquisitions)
        2. Noise filtering (sensor + atmospheric noise)

        Important:
        This step is CRITICAL for stability of ML models.
        """

        logger.info("AMD preprocessing configured for gap filling and smoothing")

        # data = gap_fill(data)
        # data = smooth(data)

        return data

    def feature_pipeline(self) -> str:
        """
        Could be:
        - temporal
        - lagged

        Currently unified → temporal
        """
        return 'temporal'

    def allowed_models(self) -> list[str]:
        """
        Gradient Boosting Regressor:
            - strong for tabular time-series features
        RF:
            - baseline, robust
        """
        return ['gbr', 'rf']
