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

    NEXT:

    velocity residual
        │
        ├── CUSUM+  acceleration
        ├── CUSUM−  deceleration
        └── VAR     oscillation
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

    def run(self, z: np.ndarray):
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

        """"
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
        """
        s_pos = np.zeros_like(z)  # acceleration
        s_neg = np.zeros_like(z)  # deceleration

        alarm_acc = np.zeros_like(z, dtype=bool)
        alarm_dec = np.zeros_like(z, dtype=bool)

        for t in range(1, len(z)):
            if np.isnan(z[t]):
                s_pos[t] = s_pos[t - 1]
                s_neg[t] = s_neg[t - 1]
                continue

            # upward drift: failure acceleration
            s_pos[t] = max(0, s_pos[t - 1] + z[t] - self.k)

            # downward drift: stabilization
            s_neg[t] = max(0, s_neg[t - 1] - z[t] - self.k)

            if s_pos[t] > self.h:
                alarm_acc[t] = True

            if s_neg[t] > self.h:
                alarm_dec[t] = True

        return s_pos, s_neg, alarm_acc, alarm_dec
