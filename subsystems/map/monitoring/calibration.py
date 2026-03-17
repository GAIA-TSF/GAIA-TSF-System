import numpy as np



def calibrate_cusum(residuals, calibration_end):
    """
    Autocorrelation-aware baseline estimation for geophysical time series.
    Converts classical CUSUM -> deformation instability detector.
    """

    # -----------------------------
    # 1) Clean baseline
    # -----------------------------
    baseline = residuals[:calibration_end]
    baseline = baseline[~np.isnan(baseline)]

    # require variance stability instead of large sample count
    if len(baseline) < 6:
        raise RuntimeError(
            f"Calibration window too short ({len(baseline)} samples). "
            "Increase calibration_fraction."
        )

    # variance must be non-degenerate
    if np.nanstd(baseline) < 1e-8:
        raise RuntimeError(
            "Calibration signal has near-zero variance — no stable regime detected."
        )

    # -----------------------------
    # 2) Mean
    # -----------------------------
    mu0 = np.mean(baseline)

    # -----------------------------
    # 3) Lag-1 autocorrelation
    # -----------------------------
    x0 = baseline[:-1]
    x1 = baseline[1:]

    # handle constant signal
    if np.std(x0) == 0 or np.std(x1) == 0:
        rho = 0.0
    else:
        rho = np.corrcoef(x0, x1)[0, 1]

    # numerical safety
    rho = np.clip(rho, -0.95, 0.95)

    # -----------------------------
    # 4) Effective sample size
    # -----------------------------
    n = len(baseline)
    n_eff = n * (1 - rho) / (1 + rho)

    if n_eff < 5:
        raise RuntimeError(
            f"Not enough independent baseline samples (N_eff={n_eff:.2f}). "
            "Increase calibration window."
        )

    # -----------------------------
    # 5) Corrected variance
    # -----------------------------
    # inflate sigma for correlated signal
    sigma_classic = np.std(baseline)
    sigma0 = sigma_classic * np.sqrt((1 + rho) / (1 - rho))

    if sigma0 == 0:
        sigma0 = 1e-6

    # -----------------------------
    # 6) Normalized detector parameters
    # -----------------------------
    # k: detectable acceleration (~0.5σ trend)
    # h: persistence of deformation energy

    k = 0.5
    h = 5.0

    print(f"[Calibration] N={n}  N_eff={n_eff:.1f}  rho={rho:.3f}")

    return mu0, sigma0, k, h