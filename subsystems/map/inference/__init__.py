# from .predictor import Predictor

from subsystems.map.inference.predictor import Predictor
from subsystems.map.dataset.insar import create_synthetic_insar_dataset

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
        mc_samples: int = 40,
        sigma_threshold: float = 2.5,
    ):
        print("Creating predictor with MC:", mc_samples) 
        
        return Predictor(
            model=model,
            device=device,
            look_back=look_back,
            horizon=horizon,
            mc_samples=mc_samples, 
            sigma_threshold=sigma_threshold,
        )
