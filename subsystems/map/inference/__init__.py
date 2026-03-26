# from .predictor import Predictor

from subsystems.map.inference.predictor import Predictor


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

    @staticmethod
    def create_predictor(
        model,
        device,
        # model geometry
        look_back: int,
        horizon: int,
        # probabilistic inference
        mc_samples: int,
        sigma_threshold: float,
        # monitoring logic
        warmup_factor: int,
        calibration_fraction: float,
        persistence: int,
        use_model_uncertainty: bool,
    ):
        return Predictor(
            model=model,
            device=device,
            # model geometry
            look_back=look_back,
            horizon=horizon,
            mc_samples=mc_samples,
            sigma_threshold=sigma_threshold,
            warmup_factor=warmup_factor,
            calibration_fraction=calibration_fraction,
            persistence=persistence,
            use_model_uncertainty=use_model_uncertainty,
        )
