from typing import Any

from core.registry import register_variable
from core.interfaces import VariablePlugin


@register_variable('slope')
class SlopeVariable(VariablePlugin):
    name = 'slope'

    def preprocess(self, data: Any, config: Any) -> Any:
        """
        InSAR displacement:
        - usually already filtered
        - optionally: detrending, normalization

        Keep minimal to avoid over-processing.
        """
        return data

    def feature_pipeline(self) -> str:
        """
        Temporal features:
        - rolling stats
        - velocity / acceleration
        - seasonal indicators
        """
        return 'temporal'

    def allowed_models(self) -> list[str]:
        """
        LSTM → captures temporal dynamics
        RF → captures nonlinear patterns in engineered features
        """
        return ['lstm', 'rf']
