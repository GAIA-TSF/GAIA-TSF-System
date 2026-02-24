
from subsystems.map.monitoring.calibration import calibrate_cusum
from subsystems.map.monitoring.cusum import CUSUMDetector
import numpy as np


def _compute_monitoring_regions(n_residuals, predictor, monitor_cfg):
    """
    Define monitoring phases in residual time domain.
    Warmup depends on model memory, not dataset size.
    """

    # -----------------------------
    # 1) WARMUP: model stabilization period
    # -----------------------------
    model_memory = predictor._look_back + predictor._horizon
    warmup_end = int(model_memory * monitor_cfg["warmup_factor"])

    warmup_end = min(warmup_end, n_residuals // 2)

    # -----------------------------
    # 2) CALIBRATION: learn baseline noise
    # -----------------------------
    calibration_length = int(n_residuals * monitor_cfg["calibration_fraction"])
    calibration_end = warmup_end + calibration_length

    calibration_end = min(calibration_end, n_residuals - 5)

    print("[Monitoring]")
    print(" Warmup end:      ", warmup_end)
    print(" Calibration end: ", calibration_end)
    print(" Monitoring start:", calibration_end)

    return warmup_end, calibration_end 


def run_monitoring(residuals, std_pred, predictor, monitor_cfg):

    n = len(residuals)

    # --------------------------------------------------
    # 1) Define monitoring phases
    # --------------------------------------------------
    # warmup_end, calibration_end = _compute_monitoring_regions(n, monitor_cfg)
    warmup_end, calibration_end = _compute_monitoring_regions(n, predictor, monitor_cfg)

    # --------------------------------------------------
    # 2) Estimate baseline ONLY from calibration region
    # --------------------------------------------------
    # --- velocity residual ---
    vel_residuals = np.diff(residuals, prepend=residuals[warmup_end])

    calibration_vel = vel_residuals[warmup_end:calibration_end]

    mu0, sigma0, k, h = calibrate_cusum(calibration_vel, len(calibration_vel))

    print(f"[Baseline μ] {mu0:.4f}")
    print(f"[Baseline σ] {sigma0:.4f}")
    print(f"[CUSUM normalized] k={k:.2f}  h={h:.2f}")

    # --------------------------------------------------
    # 3) Run CUSUM on full residual series
    # --------------------------------------------------
    # --- normalize velocity residuals ---
    z = (vel_residuals - mu0) / sigma0

    cusum = CUSUMDetector(k, h)

    # allocate full-length outputs
    S = np.zeros_like(residuals)
    alarms = np.zeros_like(residuals, dtype=bool)

    # run ONLY during monitoring
    S_monitor, alarms_monitor = cusum.run(z[calibration_end:])

    # place back into global timeline
    S[calibration_end:] = S_monitor
    alarms[calibration_end:] = alarms_monitor
    
    # --------------------------------------------------
    # 4) Model-based anomaly magnitude (optional)
    # --------------------------------------------------
    D, threshold, anomaly_mask = predictor.detect_anomaly(residuals, std_pred)

    # also ignore pre-monitoring anomalies
    anomaly_mask[:calibration_end] = False

    # --------------------------------------------------
    # 5) Build monitoring result dictionary
    # --------------------------------------------------
    monitoring = {
        "S": S,
        "alarms": alarms,
        "D": D,
        "threshold": threshold,
        "anomaly_mask": anomaly_mask,
        "h": h,
        "warmup_end": warmup_end,
        "calibration_end": calibration_end,
        "monitor_start": calibration_end,
        "baseline_sigma": sigma0,
        "baseline_mean": mu0, 
    }

    return monitoring
