import numpy as np
import torch
import torch.nn as nn 

"""
This class runs sliding-window inference and reconstructs 
a continuous prediction series.
"""


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
            self.model = model.to(device)
            self.device = device

            self.look_back = look_back
            self.horizon = horizon

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


    # Enable MC dropout during inference
    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _compute_monitoring_regions(self, series_length):

        warmup = self._warmup_factor * self.look_back
        calibration = int(self._calibration_fraction * series_length)

        monitor_start = warmup + calibration

        self._monitor_start = monitor_start

        print('[Monitoring]')
        print(f' Warmup end:       {warmup}')
        print(f' Calibration end:  {monitor_start}')
        print(f' Monitoring start: {monitor_start}')
    
    def _fit_baseline(self, residuals):

        warmup = self._warmup_factor * self.look_back
        calib_end = self._monitor_start

        baseline = residuals[warmup:calib_end]

        self._baseline_sigma = np.nanstd(baseline)

        print(f'[Baseline] residual σ = {self._baseline_sigma:.4f}')


    def predict_series(
        self,
        displacement: np.ndarray,
    ):        
        """Predict entire series using sliding windows."""

        mean_pred = np.full(len(displacement), np.nan)
        std_pred = np.full(len(displacement), np.nan)

        for i in range(len(displacement) - self.look_back - self.horizon):

            window = displacement[i : i + self.look_back]

            inputs = (
                torch.tensor(window, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )

            # Monte Carlo sampling
            samples = []

            for _ in range(self._mc_samples):
                self._enable_dropout()
                with torch.no_grad():
                    forecast = self.model(inputs).cpu().numpy().flatten()
                samples.append(forecast)

            samples = np.stack(samples)

            mean_forecast = samples.mean(axis=0)
            std_forecast = samples.std(axis=0)
            
            # add minimum uncertainty (noise floor - HARD CODED) 
            # Why? Because even a perfect model cannot predict measurement noise.
            # Without this, probabilistic detection is meaningless.
            noise_floor = 0.35  # ~ measurement noise of InSAR-like signal
            std_forecast = np.maximum(std_forecast, noise_floor)

            idx = i + self.look_back

            # average overlapping predictions with accumulation 
            counts = np.zeros(len(displacement))
            mean_sum = np.zeros(len(displacement))
            var_sum = np.zeros(len(displacement)) 

            for j in range(self.horizon):
                t = idx + j
                if t < len(displacement):
                    mean_sum[t] += mean_forecast[j]
                    var_sum[t] += std_forecast[j] ** 2
                    counts[t] += 1

            valid = counts > 0
            mean_pred[valid] = mean_sum[valid] / counts[valid]
            std_pred[valid] = np.sqrt(var_sum[valid] / counts[valid])

        return mean_pred, std_pred

    def anomaly_score(self, residuals):
        residuals = np.asarray(residuals)

        # === calibration phase ===  
        n_calib = int(len(residuals) * self._calibration_fraction)
        calib_res = residuals[:n_calib]

        mean = np.mean(calib_res)
        std = np.std(calib_res) + 1e-8

        z = (residuals - mean) / std

        # optional persistence smoothing
        if self._persistence > 1:
            z_smoothed = np.convolve(
                np.abs(z),
                np.ones(self._persistence) / self._persistence,
                mode='same'
            ) 
        else:
            z_smoothed = np.abs(z)

        return z_smoothed

    @staticmethod
    def compute_residuals(
        observations: np.ndarray,
        predictions: np.ndarray,
    ):
        residuals = observations - predictions
        return residuals
    
    @staticmethod
    def compute_velocity_residuals(
        observations: np.ndarray,
        predictions: np.ndarray,
    ):
        """
        Velocity residuals (first derivative mismatch)

        v_obs(t)  = x(t) - x(t-1)
        v_pred(t) = x̂(t) - x̂(t-1)

        Detects acceleration instead of displacement offset.
        """

        obs_v = np.diff(observations, prepend=observations[0])
        pred_v = np.diff(predictions, prepend=predictions[0])

        return obs_v - pred_v

    def detect_anomaly(self, residuals, std_pred):

        dd = np.abs(residuals) 

        # reduce horizon dimension
        if dd.ndim > 1:
            dd = np.mean(dd, axis=0)  

        # threshold selection
        if self._use_model_uncertainty:
            threshold = self._sigma_threshold * std_pred
        # else:
        #     threshold = np.full_like(residuals,
        #                             self._sigma_threshold * self._baseline_sigma)

        else:
            if self._baseline_sigma is None:
                raise RuntimeError('Baseline sigma not initialized')
            threshold = np.full_like(
                residuals,
                self._sigma_threshold * self._baseline_sigma,
            )

        if threshold.ndim > 1:
            threshold = np.mean(threshold, axis=0) 

        anomaly_mask = dd > threshold

        # disable detection before monitoring
        # anomaly_mask[:self._monitor_start] = False

        # keep NaNs from breaking persistence 
        anomaly_mask[np.isnan(dd)] = False

        # persistence rule
        for i in range(len(anomaly_mask)):
            if anomaly_mask[i]:
                window = anomaly_mask[i:i+self._persistence]
                if np.sum(window) < self._persistence:
                    anomaly_mask[i] = False

        return dd, threshold, anomaly_mask
