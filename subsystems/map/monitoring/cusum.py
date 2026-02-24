
import numpy as np

"""
Updated CUSUM detector: 
- works on velocity residuals 
"""


class CUSUMDetector:
    """
    One-sided CUSUM for gradual acceleration detection.

    Designed to work on VELOCITY residuals:
        v_res(t) = (obs_t - obs_{t-1})
                   - (pred_t - pred_{t-1})

    Optionally expects normalized input.
    """

    def __init__(self, k: float, h: float):
        """
        Parameters
        ----------
        k : float
            Reference value (minimum detectable drift / 2).
        h : float
            Decision threshold.
        """
        self.k = k
        self.h = h

    def run(self, signal: np.ndarray):
        """
        Run one-sided positive CUSUM.

        Parameters
        ----------
        signal : np.ndarray
            Velocity residual signal (preferably normalized).

        Returns
        -------
        S : np.ndarray
            CUSUM statistic
        alarms : np.ndarray (bool)
            Alarm indicator
        """

        S = np.zeros_like(signal, dtype=float)
        alarms = np.zeros_like(signal, dtype=bool)

        for t in range(1, len(signal)):

            x = signal[t]

            if np.isnan(x):
                S[t] = S[t - 1]
                continue

            # Page CUSUM update
            S[t] = max(0.0, S[t - 1] + x - self.k)

            if S[t] > self.h:
                alarms[t] = True

        return S, alarms