
import numpy as np


class CUSUMDetector:
    """
    One-sided CUSUM for gradual acceleration detection.
    Works on residual signal R(t).
    """

    def __init__(self, k: float, h: float):
        self.k = k
        self.h = h

    def run(self, residuals: np.ndarray):
        S = np.zeros_like(residuals)
        alarms = np.zeros_like(residuals, dtype=bool)

        for t in range(1, len(residuals)):
            r = residuals[t]

            if np.isnan(r):
                S[t] = S[t-1]
                continue

            S[t] = max(0, S[t-1] + r - self.k)

            if S[t] > self.h:
                alarms[t] = True

        return S, alarms
