import numpy as np
import torch

"""
This class runs sliding-window inference and reconstructs 
a continuous prediction series.
"""


class Predictor:
    """Runs model inference over time series and computes residuals."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        look_back: int,
        horizon: int,
    ):
        self._model = model.to(device)
        self._device = device
        self._look_back = look_back
        self._horizon = horizon

    def predict_series(
        self,
        displacement: np.ndarray,
    ):
        """Predict entire series using sliding windows."""

        self._model.eval()

        predictions = np.full(len(displacement), np.nan)

        with torch.no_grad():
            for i in range(len(displacement) - self._look_back - self._horizon):
                window = displacement[i : i + self._look_back]

                inputs = (
                    torch.tensor(
                        window,
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(self._device)
                )

                forecast = self._model(inputs).cpu().numpy().flatten()

                pred_index = i + self._look_back

                predictions[pred_index : pred_index + self._horizon] = forecast

        return predictions

    @staticmethod
    def compute_residuals(
        observations: np.ndarray,
        predictions: np.ndarray,
    ):
        residuals = observations - predictions
        return residuals

    @staticmethod
    def anomaly_score(
        residuals: np.ndarray,
    ):
        """Absolute residual as anomaly magnitude."""
        return np.abs(residuals)
