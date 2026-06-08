"""
# GAIA-TSF
# EO-based Temporal Spectral Anomaly Monitoring
# TASK 1 — Exploratory Data Analysis

1. Loading Sentinel-2 image time series
2. Cloud masking
3. AMD index calculation (AMD_diff)
4. Masked extraction for: clean water, TSF water, leak water
5. Robust outlier removal (MAD-based)
6. Temporal gap filling
7. Temporal smoothing
8. Basic statistics calculation
9. JSON export
10. Temporal visualization with confidence intervals
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import rasterio
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import yaml


###--- SETUP ---### 
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# inputs / outputs 
inputs_dir = cfg["project"]["inputs_dir"]
static_dir = cfg["project"]["static_dir"]
output_dir = cfg["project"]["output_dir"]

tif_dir = os.path.join(inputs_dir, "sentinel2")

# masks 
clean_mask_path = os.path.join(
    static_dir,
    cfg["masks"]["clean_water"]
)

tsf_mask_path = os.path.join(
    static_dir,
    cfg["masks"]["tsf_water"]
)

leak_mask_path = os.path.join(
    static_dir,
    cfg["masks"]["leak_water"]
)

# CONSTANTS 
CLOUD_THRESHOLD = cfg["clouds"]["threshold"]["blue_band_value"]
MAX_CLOUD_PERCENT = cfg["clouds"]["max_cloud_percent"]
ROLLING_WINDOW = cfg["processing"]["rolling_window"]


# HELPER FUNCTIONS
def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)



def calculate_cloud_mask(img, threshold=1000):
    """
    Simple cloud mask based on Blue band threshold.
    Sentinel-2:
        B2 = index 1
    """

    blue = img[1]

    cloud_mask = (blue > threshold).astype(np.uint8)

    return cloud_mask


def calculate_scl_cloud_mask(
        img,
        cloud_classes
    ):

        scl = img[0]

        cloud_mask = np.isin(
            scl,
            cloud_classes
        ).astype(np.uint8)

        return cloud_mask


def calculate_cloud_coverage(cloud_mask):
    return 100 * np.mean(cloud_mask)


def calculate_amd_indices(img):
    """
    Sentinel-2 convention:
        B2 = Blue  = index 1
        B4 = Red   = index 3
    """

    B2 = img[1].astype(np.float32)
    B4 = img[3].astype(np.float32)

    eps = 1e-6

    amd_ratio = B4 / (B2 + eps)

    amd_diff = B4 - B2

    amwi = (B4 + B2) / (B4 - B2 + eps)

    return {
        "AMD_ratio": amd_ratio,
        "AMD_diff": amd_diff,
        "AMWI": amwi
    }


def robust_outlier_filter(df, column="value", threshold=3):
    """
    MAD-based robust outlier removal.
    """

    median = df[column].median()

    mad = np.nanmedian(np.abs(df[column] - median))

    if mad == 0:
        return df

    z_robust = 0.6745 * (df[column] - median) / mad

    df["z_robust"] = z_robust

    df = df[np.abs(df["z_robust"]) <= threshold]

    return df


def temporal_postprocessing(df, column="value"):        

    df = df.sort_index()

    # interpolate temporal gaps
    df[column] = df[column].interpolate(method="time")

    # Savitzky-Golay smoothing
    values = df[column].values

    # ensure enough samples
    if len(values) >= 7:

        smooth = savgol_filter(
            values,
            window_length=7,
            polyorder=2
        )

        df[f"{column}_smooth"] = smooth

    else:

        df[f"{column}_smooth"] = values

    return df


def compute_statistics(values):

    values = np.array(values)

    median = np.nanmedian(values)

    mad = np.nanmedian(np.abs(values - median))

    stats = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "median": float(median),
        "mad": float(mad)
    }

    return stats


# LOAD MASKS
clean_mask = load_mask(clean_mask_path)
tsf_mask = load_mask(tsf_mask_path)
leak_mask = load_mask(leak_mask_path)


# LOAD IMAGE SERIES
files = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))

records_clean = []
records_tsf = []
records_leak = []

all_records_clean = []
all_records_tsf = []
all_records_leak = []


cloud_cover = []


# PROCESS SCENES
for f in files:

    date_str = os.path.basename(f).split(".")[0]

    date = pd.to_datetime(date_str)

    print(date)

    with rasterio.open(f) as src:

        img = src.read()
    
    # CLOUD MASK    
    f_scl = f.replace("sentinel2", "sentinel2_cloud").replace(".tif", "_SCL.tif")  
    
    with rasterio.open(f_scl) as src2:

        scl = src2.read()
    
    cloud_method = cfg["clouds"]["method"]

    if cloud_method == "threshold":

        cloud_mask = calculate_cloud_mask(
            img,
            threshold=CLOUD_THRESHOLD
        )

    elif cloud_method == "scl":

        cloud_mask = calculate_scl_cloud_mask(
            scl,
            cloud_classes=cfg["clouds"]["scl"]["classes"]
        )

    else:

        raise ValueError(
            f"Unknown cloud method: {cloud_method}"
        ) 

    cloud_percent = calculate_cloud_coverage(cloud_mask)
    cloud_cover.append(cloud_percent)

    # discard cloudy scenes
    if cloud_percent > MAX_CLOUD_PERCENT:
        print(f"Discarded: {cloud_percent:.2f}% clouds")
        continue

    # AMD INDICES
    indices = calculate_amd_indices(img)

    amd = indices[
        cfg["indices"]["amd_metric"]
    ]

    # APPLY CLOUD MASK
    amd = np.where(cloud_mask == 1, np.nan, amd)

    # EXTRACT MASKED VALUES
    clean_values = amd[clean_mask == 1]
    tsf_values = amd[tsf_mask == 1]
    leak_values = amd[leak_mask == 1]

    # mean scene value
    records_clean.append({"date": date, "value": np.nanmean(clean_values)})
    records_tsf.append({"date": date, "value": np.nanmean(tsf_values)})
    records_leak.append({"date": date, "value": np.nanmean(leak_values)})
    # all pixel values
    for v in clean_values:
        if np.isnan(v):
            continue
        all_records_clean.append({
            "date": date,
            "water_type": "clean",
            "AMD_diff": float(v)
        })

    for v in tsf_values:
        if np.isnan(v):
            continue
        all_records_tsf.append({
            "date": date,
            "water_type": "tsf",
            "AMD_diff": float(v)
        })

    for v in leak_values:
        if np.isnan(v):
            continue
        all_records_leak.append({
            "date": date,
            "water_type": "leak",
            "AMD_diff": float(v)
        })


# DATAFRAMES
df_clean = pd.DataFrame(records_clean)
df_tsf = pd.DataFrame(records_tsf)
df_leak = pd.DataFrame(records_leak)

df_clean_all = pd.DataFrame(all_records_clean)
df_tsf_all = pd.DataFrame(all_records_tsf)
df_leak_all = pd.DataFrame(all_records_leak)

# print(df_tsf.shape) 
# print(df_tsf_all.head())


df_clean = df_clean.set_index("date")
df_tsf = df_tsf.set_index("date")
df_leak = df_leak.set_index("date")

df_clean_all = df_clean_all.set_index("date")
df_tsf_all = df_tsf_all.set_index("date")
df_leak_all = df_leak_all.set_index("date") 


# OUTLIER REMOVAL
df_clean_no_outliers = robust_outlier_filter(df_clean_all, column="AMD_diff", threshold=3)     
# df_clean_no_ouliers = df_clean_no_ouliers.set_index("date")

# df_tsf = robust_outlier_filter(df_tsf)
df_leak_no_outliers = robust_outlier_filter(df_leak_all, column="AMD_diff", threshold=3)


# GAP FILLING + SMOOTHING
df_clean_smooth = temporal_postprocessing(df_clean_no_outliers, column="AMD_diff")  
# df_tsf = temporal_postprocessing(df_tsf)
# df_leak = temporal_postprocessing(df_leak)


# STATISTICS
stats = {
    "clean_water": compute_statistics(df_clean["value"]),
    "tsf_water": compute_statistics(df_tsf["value"]),
    "leak_water": compute_statistics(df_leak["value"])
}


# SAVE JSON
json_path = os.path.join(output_dir, "amd_eda_statistics.json")

with open(json_path, "w") as f:
    json.dump(stats, f, indent=4)

print("Statistics saved:")
print(json_path)


### --- TEMPORAL PLOT ---### 

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 14),
    sharex=True
)

# MAIN TEMPORAL PLOT - CLEAN WATER
ax = axes[0]

mean_clean = df_clean["value"]
std_clean = df_clean["value"].std()

ax.set_ylim(-200, 2500)
ax.plot(
    df_clean.index,
    mean_clean,
    marker='o',
    label='Clean water',
    color='blue',
    alpha=0.9,
)

# ax.fill_between(
#     df_clean.index,
#     mean_clean - (3 * std_clean),
#     mean_clean + (3 * std_clean),
#     color='blue',
#     alpha=0.3
# )

# pixel observations
# ax.scatter(
#     df_clean_all.index,
#     df_clean_all["AMD_diff"],
#     s=3,
#     color='blue',
#     alpha=0.5
# )

# TSF WATER
mean_tsf = df_tsf["value"]
std_tsf = df_tsf["value"].std()

# original observations
ax.scatter(
    df_tsf_all.index,
    df_tsf_all["AMD_diff"],
    s=3,
    color='salmon',
    alpha=0.3,
    label='TSF observations'
)

# temporal mean
ax.plot(
    df_tsf.index,
    mean_tsf,
    marker='o',
    color='red',
    label='TSF water'
)

# CI
# ax.fill_between(
#     df_tsf.index,
#     mean_tsf - std_tsf,
#     mean_tsf + std_tsf,
#     color='red',
#     alpha=0.3
# )


# LEAK WATER
mean_leak = df_leak["value"]
std_leak = df_leak["value"].std()

ax.plot(
    df_leak.index,
    mean_leak,
    marker='o',
    color='orange',
    label='Leak water'
)

# ax.fill_between(
#     df_leak.index,
#     mean_leak - (3 * std_leak),
#     mean_leak + (3 * std_leak),
#     color='orange',
#     alpha=0.3
# )


# MAIN PLOT SETTINGS
ax.set_title("Temporal AMD_diff Evolution")
ax.set_ylabel("AMD_diff")
ax.grid(True)
ax.legend()


# OUTLIER FILTERING PANEL
ax2 = axes[1]
ax2.set_ylim(-100, 200)
# TEMPORAL AGGREGATION
stats_clean_all = (
    df_clean_no_outliers
    .groupby(df_clean_no_outliers.index)
    .agg({
        "AMD_diff": ["mean", "std", "count"]
    })
)

stats_clean_all.columns = [
    "mean",
    "std",
    "count"
]


# 95% CONFIDENCE INTERVAL
stats_clean_all["ci95_upper"] = (
    stats_clean_all["mean"]
    + 1.96 * (
        stats_clean_all["std"]
        / np.sqrt(stats_clean_all["count"])
    )
)

stats_clean_all["ci95_lower"] = (
    stats_clean_all["mean"]
    - 1.96 * (
        stats_clean_all["std"]
        / np.sqrt(stats_clean_all["count"])
    )
)


# SMOOTHED TEMPORAL TRENDS
#  window_length must be less than or equal to the size of x 
# CLEAN WATER
stats_clean_all["smooth"] = savgol_filter(
    stats_clean_all["mean"],
    window_length=7,
    polyorder=2
)

# filtered observations
ax2.scatter(
    df_clean_no_outliers.index,
    df_clean_no_outliers["AMD_diff"], 
    s=5,
    color='blue',
    alpha=0.2,
    label='Filtered observations'
)

# filtered temporal signal
ax2.plot(
    stats_clean_all.index,
    stats_clean_all["mean"],
    color='blue',
    linewidth=2, 
    alpha=0.3,
)

# ax2.fill_between(
#     stats_clean_all.index,
#     stats_clean_all["mean"] + stats_clean_all["std"],
#     stats_clean_all["mean"] - stats_clean_all["std"],
#     color='blue',
#     alpha=0.2
# )

# IDENTIFY REMOVED OUTLIERS
removed = df_clean.loc[
    ~df_clean.index.isin(df_clean_no_outliers.index)
]

if len(removed) > 0:

    ax2.scatter(
        removed.index,
        removed["value"],
        s=50,
        color='red',
        marker='x',
        label='Removed outliers'
    )


# SETTINGS
ax2.set_title(f"Clean Water Outlier Removal — Max.: {df_clean_no_outliers['AMD_diff'].max():.2f}") 
ax2.set_xlabel("Date")
ax2.set_ylabel("AMD_diff")
ax2.grid(True)
ax2.legend()

# LEAK PANEL 
# df_clean_no_smooth

ax3 = axes[2]
ax3.set_ylim(-100, 200)

mean_leak_smooth = (
    df_leak_no_outliers
    .groupby(df_leak_no_outliers.index)["AMD_diff"]
    .mean()
)

std_leak_smooth = (
    df_leak_no_outliers
    .groupby(df_leak_no_outliers.index)["AMD_diff"]
    .std()
)

stats_leak_all = (
    df_leak_all
    .groupby(df_leak_all.index)
    .agg({
        "AMD_diff": ["mean", "std", "count"]
    })
)

stats_leak_all.columns = [
    "mean",
    "std",
    "count"
]

#  window_length must be less than or equal to the size of x 
stats_leak_all["smooth"] = savgol_filter(
    stats_leak_all["mean"],
    window_length=7,
    polyorder=2
)

# filtered observations
ax3.scatter(
    df_leak_no_outliers.index,
    df_leak_no_outliers["AMD_diff"], 
    s=5,
    color='orange',
    alpha=0.3,
    label='Filtered observations'
)


# filtered temporal signal
ax3.plot(
    mean_leak_smooth.index,
    mean_leak_smooth,
    color='orange',
    linewidth=2
)

# ax3.fill_between(
#     mean_leak_smooth.index,
#     mean_leak_smooth - (std_leak_smooth),
#     mean_leak_smooth + (std_leak_smooth), 
#     color='orange',
#     alpha=0.2
# )

# SETTINGS
ax3.set_title("Smoothed Leak Water")
ax3.set_xlabel("Date")
ax3.set_ylabel("AMD_diff")
ax3.grid(True)
ax3.legend()


# FINALIZE
plt.tight_layout()

plot_path = os.path.join(
    output_dir,
    "amd_clean_leak_temporal_plot.png"
)

plt.savefig(plot_path, dpi=300)

plt.show()

print("Plot saved:")
print(plot_path)

# 
print("EDA finished.")
