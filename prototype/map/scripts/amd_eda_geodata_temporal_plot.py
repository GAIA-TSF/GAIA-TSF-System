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
import matplotlib.pyplot as plt


# DATA
inputs_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/inputs/'
tif_dir = os.path.join(inputs_dir, 'sentinel2')

static_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/static/'

clean_mask_path = os.path.join(static_dir, 'yxsjoberg_clean_water_mask.tif')
tsf_mask_path = os.path.join(static_dir, 'yxsjoberg_tsf_water_mask.tif')
leak_mask_path = os.path.join(static_dir, 'yxsjoberg_leakage_water_mask.tif')

output_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/results'

os.makedirs(output_dir, exist_ok=True)


# CONSTANTS 
CLOUD_THRESHOLD = 1000
MAX_CLOUD_PERCENT = 10
ROLLING_WINDOW = 3


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
    """
    Gap filling + smoothing.
    """

    df = df.sort_index()

    # interpolate temporal gaps
    df[column] = df[column].interpolate(method="time")

    # smoothing
    df[f"{column}_smooth"] = (
        df[column]
        .rolling(window=ROLLING_WINDOW, center=True)
        .mean()
    )

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

cloud_cover = []


# PROCESS SCENES
for f in files:

    date_str = os.path.basename(f).split(".")[0]

    date = pd.to_datetime(date_str)

    print(date)

    with rasterio.open(f) as src:

        img = src.read()
    
    # CLOUD MASK    
    cloud_mask = calculate_cloud_mask(img)
    cloud_percent = calculate_cloud_coverage(cloud_mask)
    cloud_cover.append(cloud_percent)

    # discard cloudy scenes
    if cloud_percent > MAX_CLOUD_PERCENT:
        print(f"Discarded: {cloud_percent:.2f}% clouds")
        continue

    # AMD INDICES
    indices = calculate_amd_indices(img)
    amd = indices["AMD_diff"]

    # APPLY CLOUD MASK
    amd = np.where(cloud_mask == 1, np.nan, amd)

    # EXTRACT MASKED VALUES
    clean_values = amd[clean_mask == 1]
    tsf_values = amd[tsf_mask == 1]
    leak_values = amd[leak_mask == 1]

    # mean scene value
    records_clean.append({"date": date, "value": np.nanmean(clean_values)})
    records_tsf.append({"date": date, "value": np.nanmean(tsf_values)})
    # records_leak.append({"date": date, "value": np.nanmean(leak_values)})
    


# DATAFRAMES
df_clean = pd.DataFrame(records_clean)
df_tsf = pd.DataFrame(records_tsf)
df_leak = pd.DataFrame(records_leak)

df_clean = df_clean.set_index("date")
df_tsf = df_tsf.set_index("date")
df_leak = df_leak.set_index("date")


# OUTLIER REMOVAL
df_clean = robust_outlier_filter(df_clean)
df_tsf = robust_outlier_filter(df_tsf)
df_leak = robust_outlier_filter(df_leak)


# GAP FILLING + SMOOTHING
df_clean = temporal_postprocessing(df_clean)
df_tsf = temporal_postprocessing(df_tsf)
df_leak = temporal_postprocessing(df_leak)


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

plt.figure(figsize=(14, 6))

# CLEAN WATER
mean_clean = df_clean["value_smooth"]
std_clean = df_clean["value"].std()

plt.plot(
    df_clean.index,
    mean_clean,
    marker='o',
    label='Clean water'
)

plt.fill_between(
    df_clean.index,
    mean_clean - std_clean,
    mean_clean + std_clean,
    alpha=0.2
)

# TSF WATER
mean_tsf = df_tsf["value_smooth"]
std_tsf = df_tsf["value"].std()

plt.plot(
    df_tsf.index,
    mean_tsf,
    marker='o',
    label='TSF water'
)

plt.fill_between(
    df_tsf.index,
    mean_tsf - std_tsf,
    mean_tsf + std_tsf,
    alpha=0.2
)


# LEAK WATER
mean_leak = df_leak["value_smooth"]
std_leak = df_leak["value"].std()

plt.plot(
    df_leak.index,
    mean_leak,
    marker='o',
    label='Leak water'
)

plt.fill_between(
    df_leak.index,
    mean_leak - std_leak,
    mean_leak + std_leak,
    alpha=0.2
)


# FINALIZE FIGURE
plt.title("Temporal AMD_diff Evolution")

plt.xlabel("Date")

plt.ylabel("AMD_diff")

plt.grid(True)

plt.legend()

plt.tight_layout()

plot_path = os.path.join(output_dir, "amd_temporal_plot.png")

plt.savefig(plot_path, dpi=300)

plt.show()

print("Plot saved:")
print(plot_path)

# 
print("EDA finished.")
