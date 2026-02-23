import numpy as np

def calibrate_cusum(residuals, calibration_end_index):
    """
    Estimate stable noise from calibration phase.
    """

    stable = residuals[:calibration_end_index]
    stable = stable[~np.isnan(stable)]

    sigma0 = np.std(stable)

    k = 0.5 * sigma0
    h = 6.0 * sigma0

    return k, h, sigma0
