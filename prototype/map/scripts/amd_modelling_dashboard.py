
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor


# CONFIG
AMD_THRESHOLD = 2.0
CUSUM_K = 0.1
CUSUM_H = 5


# DATA LOADING
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
        mask = np.abs(z) > 3

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



# FEATURE ENGINEERING
def create_features(df):
    df = df.copy()
    df["lag1"] = df["AMD_smooth"].shift(1)
    df["lag2"] = df["AMD_smooth"].shift(2)
    # df["lag3"] = df["AMD_smooth"].shift(3)
    df["ratio"] = df["lag1"] / (df["lag2"] + 1e-6)

    df["diff1"] = df["AMD_smooth"].diff(1)
    df["diff2"] = df["AMD_smooth"].diff(2)
    df["acc"] = df["diff1"].diff(1)
    
    df["rolling_mean"] = df["AMD_smooth"].rolling(3).mean()
    df["local_std"] = df["AMD_smooth"].rolling(5).std()
    
    return df.dropna()


# MODEL
def train_model(clean_feat):
    X = clean_feat[["lag1", "lag2", "ratio", "diff1", "diff2", "acc", "rolling_mean", "local_std"]] # "lag3",  "month"
    y = clean_feat["AMD_smooth"]

    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model


def predict(model, feat):
    X = feat[["lag1", "lag2", "ratio", "diff1", "diff2", "acc", "rolling_mean", "local_std"]] # , "rolling_mean", "month"
    return model.predict(X)



# UNCERTAINTY
def compute_uncertainty(model, clean_feat, amd_feat):
    y_true = clean_feat["AMD_smooth"]
    y_pred = predict(model, clean_feat)

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
            alpha=0.3,
            label="Clean water reference"
        )

    # AMD observed
    ax.plot(x, y_true, "ko", label="Observed (AMD)")
    # Model prediction
    ax.plot(x, y_pred, "b.", label="Expected (baseline)")

    # Uncertainty
    ax.fill_between(
        x,
        y_pred - 3 * uncertainty,
        y_pred + 3 * uncertainty,
        color="blue",
        alpha=0.2,
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


# MAIN PIPELINE
def main(dir_path, amd_fn, clean_fn):
    

    df_clean, df_amd = load_data(dir_path, amd_fn, clean_fn) 
    df_clean, df_amd = preprocess(df_clean, df_amd)

    clean_feat = create_features(df_clean)
    amd_feat = create_features(df_amd)

    model = train_model(clean_feat)

    y_pred = predict(model, amd_feat)
    y_true = amd_feat["AMD_smooth"]

    uncertainty = compute_uncertainty(model, clean_feat, amd_feat)

    residuals = pd.Series(y_true.values - y_pred, index=amd_feat.index)

    signal = y_true - AMD_THRESHOLD
    cusum = cusum_positive(signal, CUSUM_K)

    p_clean, p_trans, p_amd, regime = compute_regime(y_true, cusum)

    # --- CLEAN reference (STRICT alignment) ---
    clean_x = clean_feat.index
    clean_y = clean_feat["AMD_smooth"]

    clean_signal = clean_y - AMD_THRESHOLD

    cusum_clean = cusum_positive(
        pd.Series(clean_signal, index=clean_x),
        CUSUM_K
    )

    _, _, _, regime_clean = compute_regime(
        clean_y,
        cusum_clean
    )

    # Plots
    # plot_prediction(amd_feat.index, y_true, y_pred, uncertainty)
    # plot_residuals(amd_feat.index, residuals, y_true)
    # plot_cusum(cusum)
    # plot_regime(regime)

    # plot_monitoring_dashboard(
    #     x=amd_feat.index,
    #     y_true=y_true,
    #     y_pred=y_pred,
    #     uncertainty=uncertainty,
    #     residuals=residuals,
    #     cusum=cusum,
    #     regime=regime
    # )  

    # plot_feature_diagnostics(amd_feat) 

    # plot_feature_space(amd_feat)  

    plot_monitoring_dashboard_amd_clean(
        x=amd_feat.index,
        y_true=y_true,
        y_pred=y_pred,
        uncertainty=uncertainty,
        residuals=residuals,
        cusum=cusum,
        regime=regime,
        clean_x=clean_x,
        clean_y=clean_y, 
        clean_regime=None
    )  


if __name__ == "__main__":
    dir_path = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/inputs/gee/"
    amd_fn = "yxsjoberg_amd_series_2018_2025.csv"
    clean_fn = "yxsjoberg_clean_series_2018_2025.csv" 
    main(dir_path, amd_fn, clean_fn)  
