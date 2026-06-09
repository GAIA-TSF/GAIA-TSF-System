"""
This script applies AMD time series modelling to the spatial data. 

"""

import os
import glob
import yaml
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt 


# CONFIG
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# CONSTANTS  
CUSUM_K = cfg["monitoring"]["risk"]["cusum_threshold"]

CUSUM_H = cfg["monitoring"]["risk"]["cusum_h"]

W_SPECTRAL = (
    cfg["monitoring"]["risk"]["spectral_weight"]
)

W_PERSISTENCE = (
    cfg["monitoring"]["risk"]["persistence_weight"]
)

W_INSTABILITY = (
    cfg["monitoring"]["risk"]["instability_weight"]
)

W_ACCELERATION = (
    cfg["monitoring"]["risk"]["acceleration_weight"]
)

ZSCORE_THRESHOLD = (
    cfg["monitoring"]["risk"]["zscore_threshold"]
)



# IO 
def load_geotiff_series(tif_dir):

    files = sorted(
        glob.glob(
            os.path.join(tif_dir, "*.tif")
        )
    )

    data = []

    for f in files:

        date_str = (
            os.path.basename(f)
            .split(".")[0]
            .split("_")[0]
        )

        with rasterio.open(f) as src:

            img = src.read(1)

        data.append({
            "date": pd.to_datetime(date_str),
            "img": img,
            "path": f
        })

    return sorted(
        data,
        key=lambda x: x["date"]
    )

def load_mask(mask_path):

    with rasterio.open(mask_path) as src:

        return src.read(1)


def load_feature_stack(
    feature_dir
):

    files = sorted(
        glob.glob(
            os.path.join(
                feature_dir,
                "*.tif"
            )
        )
    )

    stack = []

    for f in files:

        with rasterio.open(f) as src:

            arr = src.read(1)

        stack.append(arr)

    return np.stack(stack)

def normalize_stack(x):

    p5 = np.nanpercentile(
        x,
        5
    )

    p95 = np.nanpercentile(
        x,
        95
    )

    x = (
        x - p5
    ) / (
        p95 - p5 + 1e-6
    )

    return np.clip(
        x,
        0,
        1
    )


def extract_pixel_timeseries(
    stack,
    y,
    x
):

    return pd.Series(
        stack[:, y, x]
    )


# STACKING
def stack_timeseries(data):

    stack = []

    dates = []

    for d in data:

        stack.append(
            d["img"]
        )

        dates.append(
            d["date"]
        )

    return (
        np.stack(stack),
        dates
    )


# BASELINE
def build_clean_baseline(
    amd_stack,
    clean_mask
):

    clean_pixels = amd_stack[
        :,
        clean_mask > 0
    ]

    clean_median = np.nanmedian(
        clean_pixels
    )

    clean_mad = np.nanmedian(
        np.abs(
            clean_pixels -
            clean_median
        )
    )

    return (
        clean_median,
        clean_mad
    )


# ANOMALY
def anomaly_stack(
    amd_stack,
    clean_median
):

    return (
        amd_stack -
        clean_median
    )


# ROBUST Z SCORE
def zscore_stack(
    amd_stack,
    clean_median,
    clean_mad
):

    eps = 1e-6

    z = (
        0.6745 *
        (amd_stack - clean_median)
        /
        (clean_mad + eps)
    )

    return z


# CUSUM
def cusum_stack(
    anomaly,
    k
):

    T, H, W = anomaly.shape

    S = np.zeros_like(
        anomaly,
        dtype=np.float32
    )

    for t in range(1, T):

        S[t] = np.maximum(
            0,
            S[t-1] +
            anomaly[t] -
            k
        )

    return S


# RISK
def sigmoid(x):

    x = np.clip(
        x,
        -50,
        50
    )

    return (
        1 /
        (1 + np.exp(-x))
    )

def risk_stack(
    zscore,
    cusum,
    roll_std,
    acc
):

    # spectral anomaly
    p_spec = sigmoid(
        zscore - ZSCORE_THRESHOLD
    )

    # persistence
    p_persist = np.clip(
        cusum / CUSUM_H,
        0,
        1
    )

    # instability
    p_instability = normalize_stack(
        roll_std
    )

    # acceleration
    p_acc = normalize_stack(
        np.abs(acc)
    )

    # weighted risk
#     risk = (
#         0.40 * p_spec +
#         0.25 * p_persist +
#         0.20 * p_instability +
#         0.15 * p_acc
#     )

    risk = (

        W_SPECTRAL * p_spec  # +

        # W_PERSISTENCE * p_persist +

        # W_INSTABILITY * p_instability +

        # W_ACCELERATION * p_acc

    )


    return np.clip(
        risk,
        0,
        1
    )


# SAVE
def save_geotiff_series(
    output_dir,
    template_path,
    stack,
    dates,
    prefix, 
    RASTER_CFG
):

    with rasterio.open(template_path) as src:

        meta = src.meta.copy()

    meta.update(

        driver=RASTER_CFG["driver"],

        count=1,

        dtype="float32",

        nodata=np.nan,

        compress=RASTER_CFG["compression"],

        predictor=RASTER_CFG["predictor"],

        blocksize=RASTER_CFG["blocksize"],

        overview_resampling=Resampling.average
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for i, date in enumerate(dates):

        out_path = os.path.join(
            output_dir,
            f"{prefix}_{date.strftime('%Y%m%d')}.tif"
        )

        with rasterio.open(
            out_path,
            "w",
            **meta
        ) as dst:

            dst.write(
                stack[i].astype(np.float32),
                1
            )

def plot_gaia_dashboard(
    dates,
    observed,
    anomaly,
    zscore,
    cusum,
    roll_std,
    acc,
    risk, 
    clean_median,
    clean_mad
):

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14, 14),
        sharex=True
    )

    # --------------------------------------------------
    # AMD_DIFF
    # --------------------------------------------------

    axes[0].plot(
        dates,
        observed,
        "ko-"
    )

    axes[0].set_ylabel(
        "AMD_diff"
    )

    axes[0].set_title(
        "Observed AMD Signal"
    )

    # --------------------------------------------------
    # ANOMALY
    # --------------------------------------------------

    axes[1].plot(
        dates,
        anomaly,
        color="orange"
    )

    axes[1].axhline(
        0,
        color="black",
        linestyle="--"
    )

    axes[1].set_ylabel(
        "Anomaly"
    )

    # --------------------------------------------------
    # Z SCORE
    # --------------------------------------------------

    axes[2].plot(
        dates,
        zscore,
        color="red"
    )

    axes[2].axhline(2)
    axes[2].axhline(3)
    axes[2].axhline(5)

    axes[2].set_ylabel(
        "Z-score"
    )

    # --------------------------------------------------
    # CUSUM
    # --------------------------------------------------

    axes[3].plot(
        dates,
        cusum,
        color="black"
    )

    axes[3].axhline(
        CUSUM_H,
        linestyle="--"
    )

    axes[3].set_ylabel(
        "CUSUM"
    )

    # --------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------

    axes[4].plot(
        dates,
        roll_std,
        label="roll_std"
    )

    axes[4].plot(
        dates,
        np.abs(acc),
        label="|acc|"
    )

    axes[4].legend()

    axes[4].set_ylabel(
        "Temporal"
    )

    # --------------------------------------------------
    # RISK
    # --------------------------------------------------

    axes[5].plot(
        dates,
        risk,
        color="red",
        linewidth=3
    )

    axes[5].fill_between(
        dates,
        0,
        risk,
        alpha=0.3
    )

    axes[5].set_ylim(
        0,
        1
    )

    axes[5].set_ylabel(
        "Risk"
    )

    axes[5].set_title(
        "GAIA-TSF AMD Risk"
    )

    plt.tight_layout()

    plt.show()


def plot_site_risk(
    dates,
    risk_stack
):

    risk_max = np.nanmax(
        risk_stack,
        axis=(1, 2)
    )

    risk_mean = np.nanmean(
        risk_stack,
        axis=(1, 2)
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        dates,
        risk_mean,
        label="Mean risk"
    )

    plt.plot(
        dates,
        risk_max,
        label="Max risk"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(True)

    plt.legend()

    plt.title(
        "Site-wide AMD Risk Evolution"
    )

    plt.show()


def plot_gaia_dashboard(
    dates,
    observed,
    anomaly,
    zscore,
    cusum,
    roll_std,
    acc,
    risk, 
    clean_median,
    clean_mad
):

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True) 

    # AMD_DIFF
    axes[0].plot(dates, observed, "k-", alpha=0.3, label="Observed AMD_diff") 
    axes[0].plot(dates, [clean_median * 1 for i in range(len(dates))], "r-", alpha=0.9, label="Clean median") 
    axes[0].plot(dates, [clean_median + (3 * clean_mad) for i in range(len(dates))], "r--", alpha=0.5)
    axes[0].plot(dates, [clean_median - (3 * clean_mad) for i in range(len(dates))], "r--", alpha=0.5)
    axes[0].set_ylabel("AMD_diff")
    axes[0].set_title("Observed AMD Signal")

    # ANOMALY
    anomaly_median = np.nanmedian(anomaly, axis=(1))
    anomaly_mad = np.nanmedian(np.abs(anomaly - anomaly_median[:, None, None]), axis=(1, 2)) 

    axes[1].plot(dates, anomaly_median, color="orange", label="Anomaly")    
    axes[1].plot(dates, anomaly_median + (1 * anomaly_mad), color="orange", linestyle="--", alpha=0.5) 
    axes[1].plot(dates, anomaly_median - (1 * anomaly_mad), color="orange", linestyle="--", alpha=0.5)  
    axes[1].fill_between(dates, anomaly_median - (1 * anomaly_mad), anomaly_median + (1 * anomaly_mad), color="orange", alpha=0.1)
    axes[1].axhline(0, color="black", linestyle="--" )
    axes[1].set_ylabel("Residuals") 
    axes[1].set_title("Detected AMD Anomaly")

    # RISK
    risk_median = np.nanmedian(risk, axis=(1))
    risk_mad = np.nanmedian(np.abs(risk - risk_median[:, None, None]), axis=(1, 2)) 
    risk_upper = risk_median + (1 * risk_mad)
    risk_lower = risk_median - (1 * risk_mad)
    np.clip(risk_upper, 0, 1, out=risk_upper) 
    np.clip(risk_lower, 0, 1, out=risk_lower)

    # axes[2].plot(dates, risk, "r", alpha=0.2, linewidth=2, label="AMD Risk") 
    axes[2].plot(dates, risk_median, "r-", alpha=0.9, linewidth=3, label="Median Risk")
    axes[2].plot(dates, risk_upper, "r--", alpha=0.5, label="Risk MAD upper")
    axes[2].plot(dates, risk_lower, "r--", alpha=0.5, label="Risk MAD lower")
    axes[2].fill_between(dates, risk_lower, risk_upper, color="red", alpha=0.1)
    axes[2].set_ylabel("Risk") 
    axes[2].set_title("AMD Spectral Risk")

    plt.tight_layout()
    plt.show()



# MAIN
def main():

    # INPUTS    
    results_dir = cfg["project"]["output_dir"]
    amd_dir = os.path.join(results_dir, "amd_indicators")
    indicator_name = (cfg["indices"]["amd_metric"]).lower() 
    tif_dir = os.path.join(amd_dir, indicator_name) 
    # print(tif_dir) 
    temporal_feature_dir = os.path.join(results_dir, 'temporal_features', indicator_name) 
    
    static_dir = cfg["project"]["static_dir"] 
    clean_mask_path = os.path.join(static_dir, cfg["masks"]["clean_water"])
    tsf_mask_path = os.path.join(static_dir, cfg["masks"]["tsf_water"])

    monitoring_dir = cfg["project"]["output_dir"].replace('results', 'monitoring')

    # LOAD
    print("[INFO] Loading data..." )

    data = load_geotiff_series(tif_dir)
    amd_stack, dates = (stack_timeseries(data))

    print("[INFO] Loading temporal features..." )
    # smooth/ lag1/ lag2/ (lag3/) diff1/ diff2/ acc/ roll_mean/ roll_std/

    smooth_stack = load_feature_stack(os.path.join(temporal_feature_dir, "smooth")) 
    lag1_stack = load_feature_stack(os.path.join(temporal_feature_dir, "lag1")) 
    lag2_stack = load_feature_stack(os.path.join(temporal_feature_dir, "lag2")) 
    diff1_stack = load_feature_stack(os.path.join(temporal_feature_dir, "diff1"))
    diff2_stack = load_feature_stack(os.path.join(temporal_feature_dir, "diff2"))
    acc_stack = load_feature_stack(os.path.join(temporal_feature_dir, "acc")) 
    roll_mean_stack = load_feature_stack(os.path.join(temporal_feature_dir, "roll_mean"))
    roll_std_stack = load_feature_stack(os.path.join(temporal_feature_dir, "roll_std"))

    clean_mask = load_mask(clean_mask_path) 

    # BASELINE
    print("[INFO] Building clean baseline...") 

    clean_median, clean_mad = (build_clean_baseline(amd_stack, clean_mask)) 
    print("Clean median:", clean_median) 
    print("Clean MAD:", clean_mad)  

    # ANOMALY
    anomaly = anomaly_stack(amd_stack, clean_median)  

    # Z SCORE
    zscore = zscore_stack(
        amd_stack,
        clean_median,
        clean_mad
    )

    # CUSUM
    print("[INFO] Computing CUSUM...") 

    cusum = cusum_stack(anomaly, CUSUM_K) 

    # RISK
    print("[INFO] Computing risk...")   
    risk = risk_stack(zscore, cusum, roll_std_stack, acc_stack) 
    risk.shape

    tsf_mask = load_mask(tsf_mask_path)

    y, x = np.where(tsf_mask > 0) 

    # y = y[0]
    # x = x[0]
    
    # aggreation for plotting 
    observed_tsf = np.nanmedian(
        amd_stack[:, y, x],
        axis=1
    )
    anomaly_tsf = np.nanmedian(
        anomaly[:, y, x],
        axis=1
    )
    risk_tsf = np.nanmedian(
        risk[:, y, x],
        axis=1
    )


    plot_gaia_dashboard(

        dates,

        observed=
            amd_stack[:, y, x],

        anomaly=
            anomaly[:, y, x],

        zscore=
            zscore[:, y, x],

        cusum=
            cusum[:, y, x],

        roll_std=
            roll_std_stack[:, y, x],

        acc=
            acc_stack[:, y, x],

        risk=
            risk[:, y, x], 

        clean_median = clean_median,

        clean_mad = clean_mad 
    )

    # plot_site_risk(dates, risk) 


    # SAVE RESULTS
    template = data[0]["path"]
    RASTER_CFG = cfg["raster"]

    print(
        "[INFO] Saving outputs..."
    )

    save_geotiff_series(
        monitoring_dir,
        template,
        anomaly,
        dates,
        "anomaly", 
        RASTER_CFG
    )

    save_geotiff_series(
        monitoring_dir,
        template,
        zscore,
        dates,
        "zscore", 
        RASTER_CFG
    )

    save_geotiff_series(
        monitoring_dir,
        template,
        cusum,
        dates,
        "cusum", 
        RASTER_CFG
    )

    save_geotiff_series(
        monitoring_dir,
        template,
        risk,
        dates,
        "risk", 
        RASTER_CFG
    )

    print(
        "[INFO] Finished."
    )

if __name__ == "__main__":

    main()
