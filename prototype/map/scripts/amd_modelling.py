"""
Temporary script to model AMD behaviour
--- 
Monitor: 'deviation from expected geochemical equilibrium'

Plot 1 — Prediction + uncertainty 
- Uncertainty = expected variability of AMD under normal conditions e.g. uncertainty = residuals.rolling(20).std() 

Plot 2 — Residuals + threshold 
residual = observed - expected 
 Residual | Meaning               
 small    | normal water           
 moderate | possible contamination 
 large    | strong AMD event       

Plot 3 — CUSUM for persistent contamination 

Plot 4 — Regime probabilities over time + Rainfall events 
Bayesian Regime Change Detection:
Regime 1: clean water
Regime 2: transitional
Regime 3: AMD-dominated

- AMD often shows:
    - episodic spikes (rainfall events)
    - oscillations (flush–recovery cycles)    
 Pattern     | Interpretation        
 stable      | clean system          
 oscillatory | unstable geochemistry 
 trending    | worsening AMD         

TODO later: 
Event-driven analysis 
- rainfall → flushing → spike in AMD_ratio (Does anomaly follow rainfall event?) 
- Risk scoring
- Spatial generalization (conceptual step) 
"""


import os
import json
import numpy as np
import pandas as pd
import matplotlib

# Non-interactive backend (recommended for batch processing)
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# CONSTANTS 
AMD_THRESHOLD = 2  

# =========================================================
# PATHS
# =========================================================

# Linux
proj_dir = "/home/lukas/ownCloud/Projects/GAIA_TSF/tsf_experiments/"

# macOS
# proj_dir = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/"

input_dir = os.path.join(
    proj_dir,
    "AMD_monitoring_Yxsjoberg/inputs/gee/"
)

res_dir = os.path.join(
    proj_dir,
    "AMD_monitoring_Yxsjoberg/results/modelling/"
)

os.makedirs(res_dir, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df_clean = pd.read_csv(
    # os.path.join(input_dir, "yxsjoberg_clean_series.csv"),
    os.path.join(input_dir, "yxsjoberg_clean_series_2018_2025.csv"),
    parse_dates=["system:time_start"]
)

df_amd = pd.read_csv(
    # os.path.join(input_dir, "yxsjoberg_amd_series.csv"),
    os.path.join(input_dir, "yxsjoberg_amd_series_2018_2025.csv"),
    parse_dates=["system:time_start"]
)

# Rename date column
df_clean.rename(
    columns={"system:time_start": "Date"},
    inplace=True
)

df_amd.rename(
    columns={"system:time_start": "Date"},
    inplace=True
)

# Set datetime index
df_clean = df_clean.set_index("Date")
df_amd = df_amd.set_index("Date")

# Sort
df_clean = df_clean.sort_index()
df_amd = df_amd.sort_index()

# =========================================================
# ROBUST OUTLIER FILTERING
# =========================================================

def robust_filter(
    df,
    variable="AMD",
    z_threshold=3,
    physical_max=None
):

    df = df.copy()

    median = np.nanmedian(df[variable])

    mad = np.nanmedian(
        np.abs(df[variable] - median)
    )

    # Avoid division by zero
    if mad == 0:
        mad = 1e-6

    df["z_robust"] = (
        0.6745 * (df[variable] - median) / mad
    )

    mask = np.abs(df["z_robust"]) <= z_threshold

    if physical_max is not None:
        mask &= df[variable] <= physical_max

    return df[mask]


# Clean water
df_clean = robust_filter(
    df_clean,
    z_threshold=3
)

# AMD water
df_amd = robust_filter(
    df_amd,
    z_threshold=6,
    physical_max=500
)

# =========================================================
# GAP FILLING + SMOOTHING
# =========================================================

# Temporal interpolation
df_clean["AMD"] = df_clean["AMD"].interpolate(
    method="time"
)

df_amd["AMD"] = df_amd["AMD"].interpolate(
    method="time"
)

# Rolling smoothing
window_smooth = 3

df_clean["AMD_smooth"] = (
    df_clean["AMD"]
    .rolling(
        window=window_smooth,
        center=True,
        min_periods=1
    )
    .mean()
)

df_amd["AMD_smooth"] = (
    df_amd["AMD"]
    .rolling(
        window=window_smooth,
        center=True,
        min_periods=1
    )
    .mean()
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(df):

    df = df.copy()

    # -----------------------------------------------------
    # TEMPORAL MEMORY
    # -----------------------------------------------------
    df["lag1"] = df["AMD_smooth"].shift(1)
    df["lag2"] = df["AMD_smooth"].shift(2)

    # -----------------------------------------------------
    # TEMPORAL DERIVATIVES
    # -----------------------------------------------------
    df["diff1"] = df["AMD_smooth"].diff(1)

    # -----------------------------------------------------
    # LOCAL TEMPORAL STATISTICS
    # -----------------------------------------------------
    df["rolling_mean"] = (
        df["AMD_smooth"]
        .rolling(3)
        .mean()
    )

    df["rolling_std"] = (
        df["AMD_smooth"]
        .rolling(5)
        .std()
    )

    # -----------------------------------------------------
    # CYCLICAL SEASONALITY
    # -----------------------------------------------------
    doy = df.index.dayofyear

    df["sin_doy"] = np.sin(
        2 * np.pi * doy / 365.25
    )

    df["cos_doy"] = np.cos(
        2 * np.pi * doy / 365.25
    )

    return df.dropna()


# Create features
clean_feat = create_features(df_clean)
amd_feat = create_features(df_amd)

# =========================================================
# MODEL TRAINING
# =========================================================

feature_cols = [
    "lag1",
    "lag2",
    "diff1",
    "rolling_mean",
    "rolling_std",
    "sin_doy",
    "cos_doy"
]

X_clean = clean_feat[feature_cols]
y_clean = clean_feat["AMD_smooth"]

# Robust tree model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)

model.fit(X_clean, y_clean)

# =========================================================
# PREDICT AMD DYNAMICS
# =========================================================

X_amd = amd_feat[feature_cols]

y_true = amd_feat["AMD_smooth"]

y_pred = model.predict(X_amd)

# =========================================================
# UNCERTAINTY ESTIMATION
# =========================================================

# Residuals on clean baseline
y_clean_pred = model.predict(X_clean)

residuals_clean = y_clean - y_clean_pred

# Rolling uncertainty
uncertainty_clean = (
    pd.Series(
        residuals_clean,
        index=clean_feat.index
    )
    .rolling(
        20,
        center=True,
        min_periods=5
    )
    .std()
)

# -----------------------------------------
# REMOVE DUPLICATE TIMESTAMPS
# -----------------------------------------
uncertainty_clean = uncertainty_clean.groupby(
    uncertainty_clean.index
).mean()

# -----------------------------------------
# GLOBAL FALLBACK
# -----------------------------------------
global_unc = np.nanmedian(
    uncertainty_clean
)

uncertainty_clean = uncertainty_clean.fillna(
    global_unc
)

# -----------------------------------------
# MAP TO AMD TIMELINE
# -----------------------------------------
uncertainty_amd = uncertainty_clean.reindex(
    amd_feat.index,
    method="nearest"
)

# Safety floor
MIN_UNCERTAINTY = 0.15

uncertainty_amd = uncertainty_amd.clip(
    lower=MIN_UNCERTAINTY
)


# =========================================================
# CONFIDENCE INTERVALS
# =========================================================

k = 3

upper = y_pred + k * uncertainty_amd
lower = y_pred - k * uncertainty_amd

# =========================================================
# ANOMALY SCORE
# =========================================================

anomaly_score = (
    np.abs(y_true - y_pred)
    / uncertainty_amd
)

# =========================================================
# RISK CLASSES
# =========================================================

risk = pd.cut(
    anomaly_score,
    bins=[0, 1, 2, 3, np.inf],
    labels=[
        "Stable",
        "Elevated",
        "Anomalous",
        "Critical"
    ]
)

# =========================================================
# MODEL EVALUATION
# =========================================================

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

mae = mean_absolute_error(
    y_true,
    y_pred
)

r2 = r2_score(
    y_true,
    y_pred
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")

# =========================================================
# SAVE METRICS
# =========================================================

metrics = {
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2)
}

with open(
    os.path.join(res_dir, "metrics.json"),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

# =========================================================
# PLOT 1 — PREDICTION + UNCERTAINTY
# =========================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

# Observed
ax.plot(
    amd_feat.index,
    y_true,
    "k.",
    markersize=5,
    label="Observed AMD"
)

# Expected baseline
ax.plot(
    amd_feat.index,
    y_pred,
    color="blue",
    linewidth=2,
    label="Expected baseline"
)

# Confidence interval
ax.fill_between(
    amd_feat.index,
    lower,
    upper,
    color="blue",
    alpha=0.2,
    label="Uncertainty envelope"
)

ax.set_title(
    "AMD Temporal Prediction with Uncertainty",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("AMD index")

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend()

plt.tight_layout()

fig.savefig(
    os.path.join(
        res_dir,
        "amd_prediction_uncertainty.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# PLOT 2 — OBSERVED AMD INDEX + ANOMALY SCORE
# =========================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(15, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# =====================================================
# TOP PANEL — OBSERVED AMD INDEX
# =====================================================

# Clean water
ax1.plot(
    df_clean.index,
    df_clean["AMD"],
    color="darkblue",
    linewidth=2,
    alpha=0.9,
    label="Clean water"
)

# AMD affected water
ax1.plot(
    df_amd.index,
    df_amd["AMD"],
    color="darkred",
    linewidth=2,
    alpha=0.9,
    label="AMD affected water"
)

# Optional smoothing
ax1.plot(
    df_clean.index,
    df_clean["AMD_smooth"],
    color="blue",
    linewidth=1.5,
    linestyle="--",
    alpha=0.7
)

ax1.plot(
    df_amd.index,
    df_amd["AMD_smooth"],
    color="red",
    linewidth=1.5,
    linestyle="--",
    alpha=0.7
)

# AMD threshold
ax1.axhline(
    AMD_THRESHOLD,
    color="black",
    linestyle=":",
    linewidth=2,
    label=f"Threshold = {AMD_THRESHOLD}"
)

# Highlight threshold exceedance
ax1.fill_between(
    df_amd.index,
    df_amd["AMD"],
    AMD_THRESHOLD,
    where=(df_amd["AMD"] >= AMD_THRESHOLD),
    color="red",
    alpha=0.25,
    interpolate=True,
    label="Threshold exceedance"
)

# Labels
ax1.set_title(
    "Observed AMD index dynamics (B4 / B2)",
    fontsize=15,
    fontweight="bold"
)

ax1.set_ylabel(
    "AMD_score (B4 / B2)"
)

ax1.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax1.legend(
    loc="upper left"
)

# =====================================================
# BOTTOM PANEL — ANOMALY SCORE
# =====================================================

ax2.plot(
    amd_feat.index,
    anomaly_score,
    color="darkred",
    linewidth=2,
    label="Anomaly score"
)

# Risk thresholds
ax2.axhline(
    1,
    color="green",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
    label="Stable"
)

ax2.axhline(
    2,
    color="orange",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
    label="Elevated"
)

ax2.axhline(
    3,
    color="red",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
    label="Critical"
)

# Highlight anomalous periods
ax2.fill_between(
    amd_feat.index,
    anomaly_score,
    3,
    where=(anomaly_score >= 3),
    color="red",
    alpha=0.3,
    interpolate=True
)

# Labels
ax2.set_title(
    "Temporal anomaly score",
    fontsize=13,
    fontweight="bold"
)

ax2.set_ylabel(
    "Anomaly score"
)

ax2.set_xlabel(
    "Date"
)

ax2.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax2.legend(
    loc="upper left"
)

# =====================================================
# FINAL LAYOUT
# =====================================================

plt.tight_layout()

# Save figure
fig.savefig(
    os.path.join(
        res_dir,
        "amd_observed_vs_anomaly.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: amd_observed_vs_anomaly.png"
)

# =========================================================
# EXPORT RESULTS
# =========================================================

results = pd.DataFrame({

    "Date": amd_feat.index,

    "observed": y_true,

    "predicted": y_pred,

    "uncertainty": uncertainty_amd,

    "lower_ci": lower,

    "upper_ci": upper,

    "anomaly_score": anomaly_score,

    "risk": risk.astype(str)
})

results.to_csv(
    os.path.join(
        res_dir,
        "amd_prediction_results.csv"
    ),
    index=False
)

print("\nResults saved to:")
print(res_dir)
