from .predictor import Predictor

"""

The inference module must stay domain-agnostic.
It should NOT decide anomalies — only produce:
- prediction
- residual
- score

The decision threshold will belong to xAI / Early Alert detector later.
"""


class InferenceModule:
    """MAP inference module."""

    def create_predictor(
        self,
        model,
        device,
        look_back,
        horizon,
    ):
        return Predictor(
            model=model,
            device=device,
            look_back=look_back,
            horizon=horizon,
        )
