"""
This script applies AMD time series modelling to the 
spatial data. 

TODO: 
- learn model from 100 pxiels in clean water mask 
- apply to all pixels in TSF mask

Seasonal effect: 
- O3: train on normalized clean regime (robust) XXX  
- O4: seasonal + anomaly decomposition (much more robust for monitoring) 

next: 
- full raster inference (vectorized, fast)
    no loops over pixels
    works on entire image stack

finally: 
- AMD anomaly map
- CUSUM map
- regime probability map
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio 
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor


# CONFIG
AMD_THRESHOLD = 2.0
CUSUM_K = 0.1
CUSUM_H = 3

# AMD MODLE 
class AMDModel:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, feat):
        feat_scaled = self.scaler.transform(feat)
        y_scaled = self.model.predict(feat_scaled)
        return self.scaler.inverse_transform_target(y_scaled)


# READ GEOTIFF SERIES 
def load_geotiff_series(tif_dir, cloud_dir):
    files = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))

    data = []

    for f in files:
        date_str = os.path.basename(f).split(".")[0]

        cloud_f = os.path.join(
            cloud_dir,
            os.path.basename(f).replace(".tif", "_pred.tif")
        )

        with rasterio.open(f) as src:
            img = src.read()  # (bands, H, W)

        with rasterio.open(cloud_f) as src:
            cloud = src.read(1)

        data.append({
            "date": pd.to_datetime(date_str),
            "img": img,
            "cloud": cloud
        })

    return sorted(data, key=lambda x: x["date"])

def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)


def select_pixel(mask):
    # pick first valid pixel (for now)
    ys, xs = np.where(mask > 0)
    return ys[0], xs[0]


# EXTRACT TIME SERIES 
def extract_timeseries(data, y, x):
    values = []

    for d in data:
        img = d["img"]
        cloud = d["cloud"]

        # skip cloudy pixels (cloud == 1)
        if cloud[y, x] == 1:
            values.append(np.nan)
            continue

        # bands (adjust if needed!)
        B2 = img[1, y, x]
        B4 = img[3, y, x]

        amd = B4 / (B2 + 1e-6)

        values.append(amd)

    dates = [d["date"] for d in data]

    df = pd.DataFrame({
        "Date": dates,
        "AMD": values
    }).set_index("Date")

    return df

# PIXELS SAMPLING 
def sample_pixels(mask, n=100, seed=42):
    np.random.seed(seed)

    ys, xs = np.where(mask > 0)

    idx = np.random.choice(len(ys), size=min(n, len(ys)), replace=False)

    return list(zip(ys[idx], xs[idx]))

# EXTRACT MULTIPLE TIME SERIES BASED ON SAMPLING 
def extract_timeseries_multi(data, pixel_list):
    all_series = []

    for i, (y, x) in enumerate(pixel_list):

        values = []

        for d in data:
            img = d["img"]
            cloud = d["cloud"]

            if cloud[y, x] == 1:
                values.append(np.nan)
                continue

            B2 = img[1, y, x]
            B4 = img[3, y, x]

            amd = B4 / (B2 + 1e-6)
            values.append(amd)

        dates = [d["date"] for d in data]

        df = pd.DataFrame({
            "Date": dates,
            "AMD": values,
            "pixel_id": i   
        }).set_index("Date")

        all_series.append(df)

    return pd.concat(all_series)


# CSV DATA LOADING
def load_data(dir_path, amd_fn, clean_fn ):
    def _load(file):
        df = pd.read_csv(
            os.path.join(dir_path, file),
            parse_dates=["system:time_start"]
        )
        df = df.rename(columns={"system:time_start": "Date"})
        return df.set_index("Date").sort_index()

    return _load(clean_fn), _load(amd_fn)



# PREPROCESSING
def remove_outliers(df, extreme=False):
    median = df["AMD"].median()
    mad = np.nanmedian(np.abs(df["AMD"] - median))
    z = 0.6745 * (df["AMD"] - median) / mad

    if extreme:
        mask = (np.abs(z) > 6) | (df["AMD"] > 20)
    else:
        mask = np.abs(z) > 2.0

    return df.loc[~mask].copy()


def preprocess(df_clean, df_amd):
    df_clean = remove_outliers(df_clean)
    df_amd = remove_outliers(df_amd, extreme=True)

    # Interpolation
    df_clean["AMD"] = df_clean["AMD"].interpolate("time")
    df_amd["AMD"] = df_amd["AMD"].interpolate("time")

    # Smoothing
    df_clean["AMD_smooth"] = df_clean["AMD"].rolling(3, center=True).mean()
    df_amd["AMD_smooth"] = df_amd["AMD"].rolling(3, center=True).mean()

    return df_clean, df_amd

def preprocess_multi(df):

    df_list = []

    for pid, group in df.groupby("pixel_id"):

        group = remove_outliers(group)

        group["AMD"] = group["AMD"].interpolate("time")
        group["AMD_smooth"] = group["AMD"].rolling(3, center=True).mean()

        df_list.append(group)

    return pd.concat(df_list) 

class StandardScalerCustom:
    def fit(self, df, cols):
        self.mean_ = df[cols].mean()
        self.std_ = df[cols].std().replace(0, 1e-6)
        self.cols = cols
        return self

    def transform(self, df):
        df = df.copy()
        df[self.cols] = (df[self.cols] - self.mean_) / self.std_
        return df

    def inverse_transform_target(self, y):
        return y * self.std_["AMD_smooth"] + self.mean_["AMD_smooth"]

class SimpleScaler:

    def fit(self, df, FEATURE_COLS):
        self.cols = FEATURE_COLS
        self.mean_ = df[self.cols].mean()
        self.std_ = df[self.cols].std().replace(0, 1)
        return self

    def transform(self, df):
        df = df.copy()
        df[self.cols] = (df[self.cols] - self.mean_) / self.std_
        return df    

# FEATURE ENGINEERING
def create_features(df):

    df = df.copy()

    # group by pixel to avoid mixing signals
    df["lag1"] = df.groupby("pixel_id")["AMD_smooth"].shift(1)
    df["lag2"] = df.groupby("pixel_id")["AMD_smooth"].shift(2)

    df["ratio"] = df["lag1"] / (df["lag2"] + 1e-6)

    df["diff1"] = df.groupby("pixel_id")["AMD_smooth"].diff(1)
    df["diff2"] = df.groupby("pixel_id")["AMD_smooth"].diff(2)
    df["acc"] = df.groupby("pixel_id")["diff1"].diff(1)

    df["rolling_mean"] = df.groupby("pixel_id")["AMD_smooth"].rolling(3).mean().reset_index(level=0, drop=True)
    df["local_std"] = df.groupby("pixel_id")["AMD_smooth"].rolling(5).std().reset_index(level=0, drop=True)

    return df.dropna()

# OPTION 4 
def build_clean_baseline(df_clean):
    """
    Build seasonal clean baseline using DOY statistics
    """

    df = df_clean.copy()
    df["doy"] = df.index.dayofyear

    # robust seasonal statistics
    baseline = df.groupby("doy")["AMD_smooth"].median()
    upper = df.groupby("doy")["AMD_smooth"].quantile(0.7) # 0.75 0.9

    return baseline, upper


def apply_clean_baseline(baseline, upper, target_index):
    """
    Map seasonal baseline to target timeline
    """

    doy = target_index.dayofyear

    y_pred = baseline.reindex(doy).values
    upper_vals = upper.reindex(doy).values

    uncertainty = upper_vals - y_pred

    return y_pred, uncertainty



# MODEL
def train_model(clean_feat_scaled):

    X = clean_feat_scaled[
        ["lag1", "lag2", "ratio", "diff1", "diff2", "acc", "rolling_mean", "local_std"]
    ]

    y = clean_feat_scaled["AMD_smooth"]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    return model


# def predict(model, feat):
#     X = feat[["lag1", "lag2", "ratio", "diff1", "diff2", "acc", "rolling_mean", "local_std"]] # , "rolling_mean", "month"
#     return model.predict(X)


def predict(model, feat, scaler, FEATURE_COLS):
    X = feat[FEATURE_COLS]
    X_scaled = scaler.transform(feat)[FEATURE_COLS]
    return model.predict(X_scaled) 


# UNCERTAINTY
def compute_uncertainty(model, clean_feat, amd_feat, scaler, FEATURE_COLS):
    y_true = clean_feat["AMD_smooth"]
    y_pred = predict(model, clean_feat, scaler, FEATURE_COLS)

    residuals = y_true - y_pred
    unc = residuals.rolling(10, min_periods=5).std()

    unc_global = np.nanmedian(unc)
    return pd.Series(unc_global, index=amd_feat.index)



# CUSUM
def cusum_positive(signal, k):
    S = np.zeros(len(signal))
    for t in range(1, len(signal)):
        S[t] = max(0, S[t-1] + (signal.iloc[t] - k))
    return pd.Series(S, index=signal.index)



# REGIME PROBABILITY
def compute_regime(y_true, cusum):
    amd_excess = np.maximum(0, y_true - AMD_THRESHOLD)

    # p_amd_raw = 1 / (1 + np.exp(-2 * amd_excess))
    p_amd_raw = 1 / (1 + np.exp(-3 * (y_true - AMD_THRESHOLD)))
    p_amd = 0.7 * p_amd_raw + 0.3 * np.clip(cusum / CUSUM_H, 0, 1)

    # p_clean = np.exp(-y_true)
    p_clean = np.exp(-y_true / 0.8) 
    p_trans = (1 - p_clean) * (1 - p_amd)

    total = p_clean + p_trans + p_amd

    p_clean /= total
    p_trans /= total
    p_amd /= total

    regime = 0*p_clean + 1*p_trans + 2*p_amd
    regime = regime.rolling(3, center=True).mean()

    return p_clean, p_trans, p_amd, regime



# PLOTTING
def plot_feature_diagnostics(feat_df):
    import matplotlib.pyplot as plt

    features = ["lag1", "lag2", "rolling_mean"]

    fig, axes = plt.subplots(
        len(features), 1,
        figsize=(12, 8),
        sharex=True
    )

    for i, f in enumerate(features):
        ax = axes[i]

        # raw feature
        ax.plot(feat_df.index, feat_df[f], color="black", label=f)

        # rolling std (local variability)
        rolling_std = feat_df[f].rolling(10).std()
        ax.plot(feat_df.index, rolling_std, color="red", label="rolling std")

        ax.set_ylabel(f)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    plt.suptitle("Feature Stability Diagnostic (Value + Variability)")
    plt.tight_layout()
    plt.show()


def plot_prediction(x, y_true, y_pred, unc):
    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true, "ko", label="Observed")
    plt.plot(x, y_pred, "b.", label="Expected")
    plt.fill_between(x, y_pred - 2*unc, y_pred + 2*unc,
                     alpha=0.2, color="blue", label="Uncertainty")
    plt.title("Prediction with Uncertainty")
    plt.legend()
    plt.show()


def plot_residuals(x, residuals, y_true):
    plt.figure(figsize=(12, 5))
    plt.plot(x, residuals, "r-", label="Residuals")
    plt.axhline(0, color="gray")

    exceed = y_true > AMD_THRESHOLD
    persistent = exceed.rolling(3).sum() >= 2

    plt.scatter(x[persistent], residuals[persistent], color="red")
    plt.title("Residuals & Threshold")
    plt.legend()
    plt.show()


def plot_cusum(cusum):
    plt.figure(figsize=(12, 5))
    plt.plot(cusum, color="red")
    plt.axhline(CUSUM_H, linestyle="--", color="black")
    plt.title("CUSUM (Contamination)")
    plt.show()


def plot_regime(regime):
    x = regime.index
    y = regime.values

    plt.figure(figsize=(12, 4))

    for i in range(len(y)-1):
        mid = (y[i] + y[i+1]) / 2
        color = "blue" if mid < 0.5 else "orange" if mid < 1.5 else "red"

        plt.plot(x[i:i+2], y[i:i+2], color=color, linewidth=3)

    plt.axhline(0.5, linestyle="--", color="blue")
    plt.axhline(1.5, linestyle="--", color="red")

    plt.yticks([0, 1, 2], ["Clean", "Transitional", "AMD"])
    plt.title("Regime Evolution")
    plt.show()

# Feature space plot 
def plot_feature_space(feat_df):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))

    plt.scatter(
        feat_df["lag1"],
        feat_df["lag2"],
        c=feat_df.index.month,
        cmap="viridis",
        s=20
    )

    plt.xlabel("lag1")
    plt.ylabel("lag2")
    plt.title("Feature Space (colored by month)")
    plt.colorbar(label="Month")
    plt.show()


# Dashboard plot 
def plot_monitoring_dashboard_amd_clean(
    x,
    y_true,
    y_pred,
    uncertainty,
    residuals,
    cusum,
    regime,
    clean_x=None,
    clean_y=None,
    clean_regime=None, 
    AMD_THRESHOLD=2.0,
    CUSUM_H=5
):

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1, 1]}
    )


    # 1. Prediction + Uncertainty + CLEAN REFERENCE
    ax = axes[0]

    # Clean water reference
    if clean_x is not None and clean_y is not None:

        ax.plot(
            clean_x,
            clean_y,
            "o",
            color="blue",
            alpha=0.1,
            label="Clean water reference"
        )

    # AMD observed
    ax.plot(x, y_true, color="orange", label="Observed (AMD)")
    ax.plot(x, y_true, '.', color="orange", label="Observed (AMD)")  
    # Model prediction
    ax.plot(x, y_pred, "ko--", label="Expected (baseline)")

    # Uncertainty
    ax.fill_between(
        x,
        y_pred - 3 * uncertainty,
        y_pred + 3 * uncertainty,
        color="blue",
        alpha=0.1,
        label="Uncertainty"
    )

    # Threshold
    ax.axhline(AMD_THRESHOLD, color="black", linestyle="--", label="AMD threshold")
    ax.set_ylabel("AMD index")
    ax.set_title("AMD Prediction with Clean Water Reference")
    ax.legend(loc="upper right")


    # 2. Residuals
    ax = axes[1]
    ax.plot(x, np.abs(residuals), "r-", label="Residuals")
    ax.axhline(0, color="gray", linestyle=":")

    exceed = y_true > AMD_THRESHOLD
    persistent = exceed.rolling(3).sum() >= 2

    ax.scatter(
        x[persistent],
        np.abs(residuals[persistent]),
        color="red",
        label="Persistent AMD > threshold"
    )

    ax.set_ylabel("|Residual|")
    ax.set_title("Residuals")
    ax.legend(loc="upper right")


    # 3. CUSUM
    ax = axes[2]
    ax.plot(x, cusum, color="black", label="CUSUM")
    ax.axhline(CUSUM_H, color="black", linestyle="--", label="Alarm threshold")

    alarm = cusum > CUSUM_H
    alarm_persistent = alarm.rolling(3).sum() >= 2

    ax.scatter(
        x[alarm_persistent],
        cusum[alarm_persistent],
        color="red",
        label="Persistent contamination"
    )

    ax.set_ylabel("CUSUM")
    ax.set_title("CUSUM Early Warning")
    ax.legend(loc="upper right")


    # 4. Regime (continuous + colored)
    ax = axes[3]
    y = regime.values

    # --- AMD regime (colored) ---
    for i in range(len(y) - 1):
        mid = (y[i] + y[i + 1]) / 2

        if mid < 0.5:
            color = "blue"
        elif mid < 1.5:
            color = "orange"
        else:
            color = "red"

        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=3)

    # --- CLEAN regime reference (NEW) ---
    if clean_x is not None and clean_regime is not None:
        ax.plot(
            clean_x,
            clean_regime,
            color="blue",
            linestyle=":",
            linewidth=2,
            alpha=0.8,
            label="Clean regime reference"
        )

    # thresholds
    ax.axhline(0.5, color="black", linestyle="--")
    ax.axhline(1.5, color="red", linestyle="--")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Clean", " ", "AMD"])
    # ax.set_ylabel("Regime")
    ax.set_title("Continuous Regime Evolution")
    ax.legend(loc="upper right")

    # FINAL
    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


### SPATIAL VERSION ### 
def main_spatial(tif_dir, cloud_dir, clean_mask_path, tsf_mask_path):

    # 1. Load data
    data = load_geotiff_series(tif_dir, cloud_dir)

    # 2. Load masks
    clean_mask = load_mask(clean_mask_path)
    tsf_mask = load_mask(tsf_mask_path)

    # 3. Select pixels
    # clean_y, clean_x = select_pixel(clean_mask) 
    clean_pixels = sample_pixels(clean_mask, n=100)
    assert len(clean_pixels) > 0, "No clean pixels found in mask" 
    tsf_y, tsf_x = select_pixel(tsf_mask)

    print(f"Number of clean pixels sampled: {len(clean_pixels)}")
    print(f"Example clean pixels: {clean_pixels[:5]}")
    print(f"TSF pixel: {tsf_y, tsf_x}")

    # 4. Extract time series
    # df_clean = extract_timeseries(data, clean_y, clean_x)
    df_clean = extract_timeseries_multi(data, clean_pixels) 
    df_amd = extract_timeseries(data, tsf_y, tsf_x)

    # 5. Apply your existing pipeline
    df_clean = preprocess_multi(df_clean)
    _, df_amd = preprocess(df_clean.copy(), df_amd)

    # clean_feat = create_features(df_clean)
    clean_feat = create_features(df_clean) 
    # amd_feat = create_features(df_amd)
    amd_feat = create_features(df_amd.assign(pixel_id=0))

    # quick test 
    print(clean_feat["AMD_smooth"].max())
    print(amd_feat["lag1"].max())

    # columns to normalize
    feature_cols = [
        "lag1", "lag2", "ratio",
        "diff1", "diff2", "acc",
        "rolling_mean", "local_std"
    ]

    target_col = ["AMD_smooth"]

    all_cols = feature_cols + target_col

    # --- OPTION 4: CLEAN BASELINE ---

    # aggregate ALL clean pixels into one seasonal signal
    clean_series = df_clean.copy()
    clean_series["AMD_smooth"] = clean_series["AMD_smooth"]

    baseline, upper = build_clean_baseline(clean_series)

    # map to AMD timeline
    y_pred, uncertainty = apply_clean_baseline(
        baseline,
        upper,
        amd_feat.index
    )    
    
    y_true = amd_feat["AMD_smooth"] 

    residuals = pd.Series(y_true.values - y_pred, index=amd_feat.index)

    signal = y_true - AMD_THRESHOLD
    cusum = cusum_positive(signal, CUSUM_K)

    _, _, _, regime = compute_regime(y_true, cusum)

    # 6. Plot
    plot_monitoring_dashboard_amd_clean(
        x=amd_feat.index,
        y_true=y_true,
        y_pred=y_pred,
        uncertainty=uncertainty,
        residuals=residuals,
        cusum=cusum,
        regime=regime,
        clean_x=clean_feat.index,
        clean_y=clean_feat["AMD_smooth"],
        clean_regime=None, 
        AMD_THRESHOLD=AMD_THRESHOLD,
        CUSUM_H=CUSUM_H
    )


if __name__ == "__main__":
    # csv data 
    # dir_path = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/inputs/gee/"
    # amd_fn = "yxsjoberg_amd_series_2018_2025.csv"
    # clean_fn = "yxsjoberg_clean_series_2018_2025.csv" 
    # main(dir_path, amd_fn, clean_fn)  
    
    # Spatial data 
    inputs_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/inputs/'
    tif_dir = os.path.join(inputs_dir, 'sentinel2')
    cloud_dir = os.path.join(inputs_dir, 'sentinel2_clouds')

    static_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/static/'
    clean_mask_path = os.path.join(static_dir, 'yxsjoberg_clean_water_mask.tif')
    tsf_mask_path = os.path.join(static_dir, 'yxsjoberg_tsf_water_mask.tif')

    main_spatial(tif_dir, cloud_dir, clean_mask_path, tsf_mask_path) 

