
import matplotlib.pyplot as plt
import numpy as np


def _draw_monitoring_regions(ax, time, mon):
    """Draw warmup / calibration / monitoring separators"""

    # warm = mon["warmup_end"]
    # calib = mon["calibration_end"]

    # vertical separators
    # ax.axvline(time[warm], color="gray", linestyle="--", linewidth=1)
    # ax.axvline(time[calib], color="black", linestyle="--", linewidth=1.5)

    warm = min(mon["warmup_end"], len(time)-1)
    calib  = min(mon["calibration_end"], len(time)-1)

    ax.axvline(time[warm], color="gray", linestyle="--", linewidth=1)
    ax.axvline(time[calib], color="black", linestyle="--", linewidth=2)

    # region shading
    ax.axvspan(time[0], time[warm], color="gray", alpha=0.08, label="Warmup")
    ax.axvspan(time[warm], time[calib], color="orange", alpha=0.08, label="Calibration")
    ax.axvspan(time[calib], time[-1], color="green", alpha=0.05, label="Monitoring")


def plot_results(time, obs, pred, std, mon):

    plt.figure(figsize=(11, 8))

    # ---------------- Prediction ----------------
    ax1 = plt.subplot(3,1,1)

    ax1.plot(time, obs, ".", color="black", label="Observed")
    ax1.plot(time, pred, ".", color="blue", label="Predicted")

    upper = pred + mon["threshold"]
    lower = pred - mon["threshold"]
    ax1.fill_between(time, lower, upper, color="blue", alpha=0.2)

    _draw_monitoring_regions(ax1, time, mon)

    ax1.set_title("Prediction with uncertainty")
    ax1.legend(loc="upper left")


    # ---------------- Residual ----------------
    ax2 = plt.subplot(3,1,2)

    ax2.plot(time, mon["D"], color="red", label="|Residual|")
    ax2.plot(time, mon["threshold"], "--", color="black", label="Threshold")

    _draw_monitoring_regions(ax2, time, mon)

    ax2.set_title("Anomaly magnitude")
    ax2.legend(loc="upper left")


    # ---------------- CUSUM ----------------
    ax3 = plt.subplot(3,1,3)

    # Align detector index with time axis
    cusum_time = time[:len(mon["S"])]

    ax3.plot(cusum_time, mon["S"], color="purple", label="CUSUM")
    ax3.axhline(mon["h"], linestyle="--", color="black", label="CUSUM limit")

    alarm_idx = np.where(mon["alarms"])[0]
    ax3.scatter(cusum_time[alarm_idx], mon["S"][alarm_idx], color="red", label="Alarm")

    _draw_monitoring_regions(ax3, time, mon)

    ax3.set_title("CUSUM early warning")
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
