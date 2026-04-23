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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

# 1. LOAD DATA
dir_path = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/inputs/gee/"

df_clean = pd.read_csv(
    os.path.join(dir_path, "yxsjoberg_clean_series.csv"),
    parse_dates=["system:time_start"]
)

df_amd = pd.read_csv(
    os.path.join(dir_path, "yxsjoberg_amd_series.csv"),
    parse_dates=["system:time_start"]
)

# Rename and set index
df_clean.rename(columns={"system:time_start": "Date"}, inplace=True)
df_amd.rename(columns={"system:time_start": "Date"}, inplace=True)

df_clean = df_clean.set_index("Date")
df_amd = df_amd.set_index("Date")

# Sort by time
df_clean = df_clean.sort_index()
df_amd = df_amd.sort_index()


# 2. OUTLIER REMOVAL
# CLEAN WATER: remove mild outliers 
median_clean = df_clean["AMD"].median()
mad_clean = np.nanmedian(np.abs(df_clean["AMD"] - median_clean))

df_clean["z_robust"] = 0.6745 * (df_clean["AMD"] - median_clean) / mad_clean
df_clean = df_clean[np.abs(df_clean["z_robust"]) <= 3]

# AMD WATER: remove ONLY extreme outliers 
median_amd = df_amd["AMD"].median()
mad_amd = np.nanmedian(np.abs(df_amd["AMD"] - median_amd))

df_amd["z_robust"] = 0.6745 * (df_amd["AMD"] - median_amd) / mad_amd

far_threshold = 6
physical_max = 20

df_amd["outlier_far"] = np.abs(df_amd["z_robust"]) > far_threshold
df_amd["outlier_physical"] = df_amd["AMD"] > physical_max

df_amd = df_amd[~(df_amd["outlier_far"] | df_amd["outlier_physical"])]


# 3. GAP FILLING + SMOOTHING
# Interpolate missing values (cloud gaps)
df_clean["AMD"] = df_clean["AMD"].interpolate(method="time")
df_amd["AMD"] = df_amd["AMD"].interpolate(method="time")

# Smooth signal (reduce noise)
df_clean["AMD_smooth"] = df_clean["AMD"].rolling(window=3, center=True).mean()
df_amd["AMD_smooth"] = df_amd["AMD"].rolling(window=3, center=True).mean()


# 4. FEATURE ENGINEERING
def create_features(df):
    df = df.copy()

    # Lagged features (system memory)
    df["lag1"] = df["AMD_smooth"].shift(1)
    df["lag2"] = df["AMD_smooth"].shift(2)
    df["lag3"] = df["AMD_smooth"].shift(3)

    # Local statistics
    df["rolling_mean"] = df["AMD_smooth"].rolling(3).mean()

    # Seasonality
    df["month"] = df.index.month

    return df.dropna()


# Prepare datasets
clean_feat = create_features(df_clean)
amd_feat = create_features(df_amd)


# 5. TRAIN BASELINE MODEL (CLEAN WATER)
X_clean = clean_feat[["lag1", "lag2", "lag3", "rolling_mean", "month"]]
y_clean = clean_feat["AMD_smooth"]

model = GradientBoostingRegressor()
model.fit(X_clean, y_clean)


# 6. PREDICT AMD SERIES
X_amd = amd_feat[["lag1", "lag2", "lag3", "rolling_mean", "month"]]
y_true = amd_feat["AMD_smooth"]

y_pred = model.predict(X_amd)


# 7. UNCERTAINTY ESTIMATION (CLEAN BASELINE)
# Residuals on CLEAN data (represents normal variability)
y_clean_pred = model.predict(X_clean)
residuals_clean = y_clean - y_clean_pred

# Rolling uncertainty (local variability)
uncertainty_clean = pd.Series(
    residuals_clean, index=clean_feat.index
).rolling(20).std()

# Global fallback (for simplicity)
unc_global = np.nanmean(uncertainty_clean)

# Map uncertainty to AMD timeline
uncertainty_amd = pd.Series(unc_global, index=amd_feat.index)

# Confidence interval
k = 3  # ~95% confidence

upper = y_pred + k * uncertainty_amd
lower = y_pred - k * uncertainty_amd


# 8. PLOT 1 — PREDICTION + UNCERTAINTY
plt.figure(figsize=(12, 5))
# Observed AMD
plt.plot(amd_feat.index, y_true, "k.", label="Observed")
# Predicted baseline
plt.plot(amd_feat.index, y_pred, "b-", label="Expected (baseline)")
# Uncertainty band (normal variability)
plt.fill_between(
    amd_feat.index,
    lower,
    upper,
    color="blue",
    alpha=0.2,
    label="Uncertainty (clean baseline)"
)
plt.ylim(0, 7)
plt.title("AMD Prediction with Uncertainty")
plt.legend()
plt.show()
