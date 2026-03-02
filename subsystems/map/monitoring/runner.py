import numpy as np

from subsystems.map.monitoring import oscillation
from subsystems.map.monitoring.calibration import calibrate_cusum
from subsystems.map.monitoring.cusum import CUSUMDetector
# from subsystems.map.monitoring.oscillation import OscillationDetector
from subsystems.map.monitoring.regime import resolve_regime 


def _ema(signal, tau):
    alpha = 2 / (tau + 1)
    out = np.zeros_like(signal)

    out[0] = signal[0]

    for i in range(1, len(signal)):
        if np.isnan(signal[i]):
            out[i] = out[i-1]
        else:
            out[i] = alpha * signal[i] + (1 - alpha) * out[i-1]

    return out


def _interp_nan(x):
    """Linear interpolation of NaNs (edge-safe)"""
    x = x.copy()
    n = len(x)

    valid = np.isfinite(x)
    if valid.sum() < 2:
        return x

    idx = np.arange(n)
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x

def persistence(signal, win=20):
    """Measure sign stability (1 = sustained motion, 0 = oscillation)"""

    s = np.sign(signal)
    s[s == 0] = 1

    changes = np.zeros_like(signal)
    changes[1:] = np.abs(np.diff(s))

    kernel = np.ones(win) / win
    change_rate = np.convolve(changes, kernel, mode="same")

    pers = 1 - change_rate
    return np.clip(pers, 0, 1)


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
    # vel_residuals = np.diff(residuals, prepend=residuals[warmup_end])
    # vel_residuals = np.diff(residuals, prepend=np.nan)
    # --------------------------------------------
    
    ### Physics-correct ### 
    # STEP 1 — velocity innovation
    # --------------------------------------------
    residuals_clean = _interp_nan(residuals)
    vel_residuals = np.gradient(residuals_clean)

    # --------------------------------------------
    # STEP 2 — creep-scale smoothing 
    # removes atmospheric & decorrelation noise
    # --------------------------------------------
    tau_days = monitor_cfg.get("creep_tau", 30)   # 30 acquisitions ≈ 1–2 months
    vel_trend = _ema(vel_residuals, tau_days) 

    filter_delay = int(monitor_cfg.get("creep_tau", 30))

    warmup_end = max(warmup_end, filter_delay)
    calibration_end = max(calibration_end, warmup_end + 20) 

    # --------------------------------------------
    # STEP 3 — acceleration innovation
    # this is the real failure precursor
    # --------------------------------------------
    # acc_residuals = np.gradient(vel_trend)

    # calibration_vel = vel_residuals[warmup_end:calibration_end]
    # calibration_vel = calibration_vel[~np.isnan(calibration_vel)]
    # calibration_acc = acc_residuals[warmup_end:calibration_end]



    # mu0, sigma0, k, h = calibrate_cusum(calibration_vel, len(calibration_vel))
    # calibration_trend = vel_trend[warmup_end:calibration_end]
    # mu0, sigma0, k, h = calibrate_cusum(calibration_trend, len(calibration_trend))
    # calibration signal = smoothed velocity
    # calibration_signal = vel_trend[warmup_end:calibration_end]
    calibration_signal = vel_trend[warmup_end:calibration_end]
    calibration_signal = calibration_signal[np.isfinite(calibration_signal)]

    if len(calibration_signal) < 20:
        raise RuntimeError(
            f"Calibration window invalid: only {len(calibration_signal)} valid samples "
            f"(warmup={warmup_end}, calib_end={calibration_end})"
        )
    mu0, sigma0, k, h = calibrate_cusum(calibration_signal, len(calibration_signal))


    if sigma0 < 1e-6 or np.isnan(sigma0):
        raise RuntimeError(
        "CUSUM baseline variance collapsed — calibration window too short or smoothing too strong"
        )

    sigma0 = max(sigma0, 1e-6) 

    print(f"[Baseline μ] {mu0:.4f}")
    print(f"[Baseline σ] {sigma0:.4f}")
    print(f"[CUSUM normalized] k={k:.2f}  h={h:.2f}")

    # --------------------------------------------------
    # 3) Run CUSUM on full residual series
    # --------------------------------------------------
    # --- normalize velocity residuals ---
    # z = (vel_residuals - mu0) / sigma0
    z = (vel_trend - mu0) / sigma0 

    # -----------------------------
    # PHYSICAL REGIME CLASSIFICATION
    # -----------------------------
    pers = persistence(z, win=monitor_cfg.get("persist_win", 25))

    cusum = CUSUMDetector(k, h)
    S_pos, S_neg, raw_acc, raw_dec = cusum.run(z[calibration_end:])

    # classify regimes
    danger_acc, danger_dec, oscillation = resolve_regime(
        S_pos, S_neg, pers[calibration_end:], h
    )

    # osc = OscillationDetector()
    # var, alarm_osc = osc.run(z[calibration_end:])
    # var, alarm_osc = osc.run(vel_residuals[calibration_end:])


    # var_full = np.zeros_like(residuals)
    # var_full[calibration_end:] = var

    # place into full timeline
    S_acc = np.zeros_like(residuals)
    S_dec = np.zeros_like(residuals)

    alarm_acc_full = np.zeros_like(residuals, dtype=bool)
    alarm_dec_full = np.zeros_like(residuals, dtype=bool)
    alarm_osc_full = np.zeros_like(residuals, dtype=bool)

    S_acc[calibration_end:] = S_pos
    S_dec[calibration_end:] = S_neg 

    alarm_acc_full[calibration_end:] = danger_acc
    alarm_dec_full[calibration_end:] = danger_dec
    alarm_osc_full[calibration_end:] = oscillation
    
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
    # model residual anomaly
    "D": D,
    "threshold": threshold,
    "anomaly_mask": anomaly_mask,

    # acceleration
    "S_acc": S_acc,
    "alarm_acc": alarm_acc_full,

    # deceleration
    "S_dec": S_dec,
    "alarm_dec": alarm_dec_full,

    # oscillation
    "alarm_osc": alarm_osc_full,
    "persistence": pers, 

    "h": h,

    # monitoring regions
    "warmup_end": warmup_end,
    "calibration_end": calibration_end,
    "monitor_start": calibration_end,

    # baseline
    "baseline_sigma": sigma0,
    }

    return monitoring
