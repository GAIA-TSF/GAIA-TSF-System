import numpy as np
import torch
import torch.nn as nn 

"""
This class runs sliding-window inference and reconstructs 
a continuous prediction series.
"""

# print("Loaded predictor with MC Dropout")

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
            mc_samples: int,
            sigma_threshold: float,
            warmup_factor: int,
            calibration_fraction: float,
            persistence: int,
            use_model_uncertainty: bool,
        ):
            self._model = model.to(device)
            self._device = device

            self._look_back = look_back
            self._horizon = horizon

            self._mc_samples = mc_samples
            self._sigma_threshold = sigma_threshold

            # monitoring parameters
            self._warmup_factor = warmup_factor
            self._calibration_fraction = calibration_fraction
            self._persistence = persistence
            self._use_model_uncertainty = use_model_uncertainty

            # learned later
            self._monitor_start = None
            self._baseline_sigma = None
        
    # Enable dropout during inference
    def _enable_dropout(self):
        for module in self._model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _compute_monitoring_regions(self, series_length):

        warmup = self._warmup_factor * self._look_back
        calibration = int(self._calibration_fraction * series_length)

        monitor_start = warmup + calibration

        self._monitor_start = monitor_start

        print(f"[Monitoring]")
        print(f" Warmup end:       {warmup}")
        print(f" Calibration end:  {monitor_start}")
        print(f" Monitoring start: {monitor_start}")
    
    def _fit_baseline(self, residuals):

        warmup = self._warmup_factor * self._look_back
        calib_end = self._monitor_start

        baseline = residuals[warmup:calib_end]

        self._baseline_sigma = np.nanstd(baseline)

        print(f"[Baseline] residual σ = {self._baseline_sigma:.4f}")



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

    def detect_anomaly(self, residuals, std_pred):

        D = np.abs(residuals)

        # threshold selection
        if self._use_model_uncertainty:
            threshold = self._sigma_threshold * std_pred
        else:
            threshold = np.full_like(residuals,
                                    self._sigma_threshold * self._baseline_sigma)

        anomaly_mask = D > threshold

        # disable detection before monitoring
        anomaly_mask[:self._monitor_start] = False

        # persistence rule
        for i in range(len(anomaly_mask)):
            if anomaly_mask[i]:
                window = anomaly_mask[i:i+self._persistence]
                if np.sum(window) < self._persistence:
                    anomaly_mask[i] = False

        return D, threshold, anomaly_mask
