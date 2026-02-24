import numpy as np


def calibrate_cusum(residuals, calibration_end):
    """
    Estimate baseline statistics and derive normalized CUSUM parameters.
    """

    baseline = residuals[:calibration_end]
    baseline = baseline[~np.isnan(baseline)]

    if len(baseline) < 10:
        raise RuntimeError("Not enough baseline samples for CUSUM calibration")

    mu0 = np.mean(baseline)
    sigma0 = np.std(baseline)

    if sigma0 == 0:
        sigma0 = 1e-6

    # normalized CUSUM parameters (dimensionless)
    # TODO: move to config 
    k = 0.5        # detect 0.5σ shift (early acceleration)
    h = 5.0        # typical ARL ~ 500 samples

    return mu0, sigma0, k, h 
