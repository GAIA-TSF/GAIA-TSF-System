import numpy as np
import torch
import torch.nn as nn 

"""
This class runs sliding-window inference and reconstructs 
a continuous prediction series.
"""

print("Loaded predictor with MC Dropout")

class Predictor:
    """
    Runs probabilistic inference using MC Dropout.
    Produces prediction mean and uncertainty band. 
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        look_back: int,
        horizon: int,
        mc_samples: int = None,
        sigma_threshold: float = None,
    ):
        self._model = model.to(device)
        self._device = device
        self._look_back = look_back
        self._horizon = horizon 

        # safe defaults if caller forgot arguments
        self._mc_samples = 40 if mc_samples is None else mc_samples
        self._sigma_threshold = 2.5 if sigma_threshold is None else sigma_threshold
        
    # Enable dropout during inference
    def _enable_dropout(self):
        for module in self._model.modules():
            if isinstance(module, nn.Dropout):
                module.train()


    def predict_series(
        self,
        displacement: np.ndarray,
    ):        
        """Predict entire series using sliding windows."""

        mean_pred = np.full(len(displacement), np.nan)
        std_pred = np.full(len(displacement), np.nan)

        for i in range(len(displacement) - self._look_back - self._horizon):

            window = displacement[i : i + self._look_back]

            inputs = (
                torch.tensor(window, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self._device)
            )

            # Monte Carlo sampling
            samples = []

            for _ in range(self._mc_samples):
                self._enable_dropout()
                with torch.no_grad():
                    forecast = self._model(inputs).cpu().numpy().flatten()
                samples.append(forecast)

            samples = np.stack(samples)

            mean_forecast = samples.mean(axis=0)
            std_forecast = samples.std(axis=0)
            
            # add minimum uncertainty (noise floor - HARD CODED) 
            # Why? Because even a perfect model cannot predict measurement noise.
            # Without this, probabilistic detection is meaningless.
            noise_floor = 0.35  # ~ measurement noise of InSAR-like signal
            std_forecast = np.maximum(std_forecast, noise_floor)

            idx = i + self._look_back

            # old 
            # mean_pred[pred_index : pred_index + self._horizon] = mean_forecast
            # std_pred[pred_index : pred_index + self._horizon] = std_forecast

            # average overlapping predictions with accumulation 
            counts = np.zeros(len(displacement))
            mean_sum = np.zeros(len(displacement))
            var_sum = np.zeros(len(displacement)) 

            for j in range(self._horizon):
                t = idx + j
                if t < len(displacement):
                    mean_sum[t] += mean_forecast[j]
                    var_sum[t] += std_forecast[j] ** 2
                    counts[t] += 1

            valid = counts > 0
            mean_pred[valid] = mean_sum[valid] / counts[valid]
            std_pred[valid] = np.sqrt(var_sum[valid] / counts[valid])

        return mean_pred, std_pred

    @staticmethod
    def compute_residuals(
        observations: np.ndarray,
        predictions: np.ndarray,
    ):
        residuals = observations - predictions
        return residuals

    def detect_anomaly(self, 
        residuals: np.ndarray, 
        std_pred: np.ndarray, 
    ):
        """
        D = |residual|
        anomaly if D > k * sigma
        """

        D = np.abs(residuals)
        threshold = self._sigma_threshold * std_pred

        # smooth threshold 
        window = 7
        threshold = np.convolve(threshold, np.ones(window)/window, mode='same') 

        anomaly_mask = D > threshold

        # ignore warm-up region 
        warmup = self._look_back + self._horizon
        anomaly_mask[:warmup] = False

        # require persistence
        min_duration = 3

        for i in range(len(anomaly_mask)):
            if anomaly_mask[i]:
                window = anomaly_mask[i:i+min_duration]
                if np.sum(window) < min_duration:
                    anomaly_mask[i] = False

        return D, threshold, anomaly_mask 
