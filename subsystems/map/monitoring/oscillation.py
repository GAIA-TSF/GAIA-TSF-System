import numpy as np


class OscillationDetector:
    """
    Detects decorrelation / atmospheric noise
    using rolling variance of velocity residuals
    """

    def __init__(self, window=12, threshold=3.0):
        self.window = window
        self.threshold = threshold

    def run(self, z):
        var = np.full_like(z, np.nan)

        for i in range(self.window, len(z)):
            var[i] = np.var(z[i - self.window : i])

        baseline = np.nanmedian(var)
        alarms = var > self.threshold * baseline

        return var, alarms
