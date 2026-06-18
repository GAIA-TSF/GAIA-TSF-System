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
import yaml
import numpy as np
import pandas as pd
import rasterio
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import dates as matplotlib_dates
import matplotlib.dates as mdates
from datetime import datetime


###--- SETUP ---### 
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# inputs / outputs 
inputs_dir = cfg["project"]["inputs_dir"]
static_dir = cfg["project"]["static_dir"]
output_dir = os.path.join(cfg["project"]["output_dir"], 'eda') 

if not os.path.isdir(output_dir): 
    os.makedirs(output_dir)


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

    cloud_mask = (blue < threshold).astype(np.uint8)

    return cloud_mask


def calculate_scl_cloud_mask(
        scl,
        cloud_classes
    ):

        cloud_mask = np.isin(
            scl,
            cloud_classes
        ).astype(np.uint8)
        cloud_mask.shape

        cloud_mask = 1 - cloud_mask 
        # plt.imshow(cloud_mask[0, :, :], cmap="gray")
        # plt.show() 

        return cloud_mask[0, :, :]


def calculate_cloud_coverage(cloud_mask):
    return 100 * np.mean(cloud_mask)


def calculate_amd_indices(img):
    """
    Sentinel-2 convention:
        B2 = Blue  = index 1
        B4 = Red   = index 3
    TODO: calculate only the selected index based on config.yaml 
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
    print(df.head(3)) 

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
scene_files = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))

print(f"Found {len(scene_files)} Sentinel-2 images in: {tif_dir}") 


# Store cloud cover percentages
cloud_cover = []

# Iterate through scenes
for i, path in enumerate(scene_files):
    # break

    filename = os.path.basename(path)
    scene_id = filename.replace(".tif", "")
    print(f"Processing {scene_id} ({i+1}/{len(scene_files)})")

    # CLOUD MASK FILE    
    f_scl = path.replace("sentinel2", "sentinel2_cloud").replace(".tif", "_SCL.tif")      
    
    cloud_method = cfg["clouds"]["method"]

    if cloud_method == "threshold": 
        with rasterio.open(path) as src:
            img = src.read()

        cloud_mask = calculate_cloud_mask(
            img,
            threshold=CLOUD_THRESHOLD
        )

    elif cloud_method == "scl":
        with rasterio.open(f_scl) as src2:
            scl = src2.read()

        cloud_mask = calculate_scl_cloud_mask(
            scl,
            cloud_classes=cfg["clouds"]["scl"]["classes"]
        )

    else:
        raise ValueError(
            f"Unknown cloud method: {cloud_method}"
        ) 
    
    # Count cloud and non-cloud pixels
    values, counts = np.unique(cloud_mask, return_counts=True)

    # Ensure both classes exist
    count_dict = dict(zip(values, counts))

    # cloud_mask:
    # 0 = cloud
    # 1 = non-cloud
    cloud_pixels = count_dict.get(0, 0)
    total_pixels = np.sum(counts)

    # Cloud percentage
    cloud_percent = (cloud_pixels / total_pixels) * 100
    cloud_cover.append(cloud_percent)
    print(f"{scene_id}: {cloud_percent:.2f}% clouds")

# Convert to numpy array if needed
cloud_cover = np.array(cloud_cover)

# SELECT CLOUDLESS SCENES
cloud_free_indices = np.where(np.array(cloud_cover) < MAX_CLOUD_PERCENT)[0]
cloudless_scenes = [scene_files[i] for i in cloud_free_indices]
print(len(cloudless_scenes), 'scenes were kept')
print(len(scene_files) - len(cloudless_scenes), 'scenes were discarded')    



# GET PIXELs
# Clean water pixels
wy, wx = np.where(clean_mask == 1)
# TSF water pixels 
ay, ax = np.where(tsf_mask == 1)
# leak water pixels 
ly, lx = np.where(leak_mask == 1) 


# CREATE ARRAYS FOR TEMPORAL VALUES
peak_water = np.zeros((len(cloudless_scenes), len(wx)))
peak_amd = np.zeros((len(cloudless_scenes), len(ax)))
peak_leak = np.zeros((len(cloudless_scenes), len(lx)))

valid_water_pixels = []
valid_amd_pixels = []
valid_leak_pixels = [] 

amd_metric = cfg["indices"]["amd_metric"] 
cloud_method = cfg["clouds"]["method"]

# PROCESS EACH SCENE
for i, path in enumerate(cloudless_scenes):
    
    print(f"Processing scene {i+1}/{len(cloudless_scenes)}")

    filename = os.path.basename(path)
    scene_id = filename.replace(".tif", "")

    # LOAD DATA
    with rasterio.open(path, 'r') as raster:
        img = raster.read() 

        # Compute spectral index 
        amd_indices = calculate_amd_indices(img)
        # amd_indices.keys()
        amd_map = amd_indices[amd_metric] 

    # CLOUD MASK FILE    
    f_scl = path.replace("sentinel2", "sentinel2_cloud").replace(".tif", "_SCL.tif")      

    if cloud_method == "threshold": 
        with rasterio.open(path) as src:
            img = src.read()

        cloud_mask = calculate_cloud_mask(
            img,
            threshold=CLOUD_THRESHOLD
        )

    elif cloud_method == "scl":
        with rasterio.open(f_scl) as src2:
            scl = src2.read()

        cloud_mask = calculate_scl_cloud_mask(
            scl,
            cloud_classes=cfg["clouds"]["scl"]["classes"]
        )

    # APPLY CLOUD MASK
    amd_map_no_cloud = np.where(cloud_mask == 0, np.nan, amd_map)
    # plt.imshow(amd_map_no_cloud, cmap="RdBu")
    # plt.colorbar() 
    # plt.show()

    # track number of valid pixels 
    water_values = amd_map_no_cloud[wy, wx]
    amd_values = amd_map_no_cloud[ay, ax]
    leak_values = amd_map_no_cloud[ly, lx] 

    peak_water[i, :] = water_values
    peak_amd[i, :] = amd_values
    peak_leak[i, :] = leak_values 

    valid_water_pixels.append(np.sum(~np.isnan(water_values)))
    valid_amd_pixels.append(np.sum(~np.isnan(amd_values)))
    valid_leak_pixels.append(np.sum(~np.isnan(leak_values)))
    
dates = []
for path in cloudless_scenes:
    filename = os.path.basename(path)
    time = filename.replace('.tif', '')
    time = time.split('T')[0]
    datetime_object = datetime.strptime(time, '%Y%m%d')
    dates.append(datetime_object)

# print(dates)

# date2num = matplotlib_dates.date2num(dates)



# CREATE FIGURE
print("Creating temporal profile plot...")
print(f"Dates: {len(dates)}")
print(f"AMD shape: {peak_amd.shape}")
print(f"Reference shape: {peak_water.shape}")


fig_path = os.path.join(
    output_dir,
    "temporal_amd_profiles.png"
)

fig, ax = plt.subplots(figsize=(12, 6))

# date formating 
ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=1)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m')
)

# fig.autofmt_xdate()

# PLOT ALL AMD PIXEL TEMPORAL PROFILES
for j in range(peak_amd.shape[1]):

    ax.plot(
        dates,
        peak_amd[:, j],
        color='red',
        alpha=0.05,
        linewidth=0.6
    )

# PLOT ALL REFERENCE WATERBODY PROFILES
for j in range(peak_water.shape[1]):

    ax.plot(
        dates,
        peak_water[:, j],
        color='blue',
        alpha=0.02,
        linewidth=0.5
    )


# OVERLAY TEMPORAL MEANS
amd_mean = np.nanmean(peak_amd, axis=1)
water_mean = np.nanmean(peak_water, axis=1)

# PERCENTILE ENVELOPES
amd_p05 = np.nanpercentile(peak_amd, 5, axis=1)
amd_p95 = np.nanpercentile(peak_amd, 95, axis=1)

water_p05 = np.nanpercentile(peak_water, 5, axis=1)
water_p95 = np.nanpercentile(peak_water, 95, axis=1)

ax.fill_between(
    dates,
    amd_p05,
    amd_p95,
    color='red',
    alpha=0.15,
    label='AMD 5–95%'
)

ax.fill_between(
    dates,
    water_p05,
    water_p95,
    color='blue',
    alpha=0.08,
    label='Reference 5–95%'
)

# profiles
ax.plot(
    dates,
    amd_mean,
    color='darkred',
    linewidth=2.5,
    label='AMD mean'
)

ax.plot(
    dates,
    water_mean,
    color='darkblue',
    linewidth=2.5,
    label='Reference water mean'
)



# LABELS
ax.set_ylabel('(B4 - B2)', fontsize=11)

ax.set_xlabel('Date', fontsize=11)

ax.set_title(
    'Temporal AMD spectral dynamics',
    fontsize=14,
    fontweight='bold'
)


# GRID
ax.grid(
    True,
    linestyle='--',
    alpha=0.3
)


# LEGEND
ax.legend(
    loc='upper right',
    frameon=True
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)


# FINAL LAYOUT
plt.tight_layout()

# SAVE FIGURE
plt.savefig(
    fig_path,
    dpi=300,
    bbox_inches='tight'
)

print(f'Figure saved to: {fig_path}')
plt.show()
plt.close()  


### --- HISTOGRAM OF AMD vs CLEAN WATER --- ###

# OUTPUT FIGURE PATH
hist_path = os.path.join(
    output_dir,
    'histogram_amd_vs_clean_water.png'
)


# FLATTEN ARRAYS
amd_values = peak_amd.flatten()
water_values = peak_water.flatten()

# REMOVE NaNs
amd_values = amd_values[~np.isnan(amd_values)]
water_values = water_values[~np.isnan(water_values)]


# CREATE FIGURE
fig, ax = plt.subplots(figsize=(10, 6))


# HISTOGRAMS
bins = 100

ax.hist(
    water_values,
    bins=bins,
    density=True,
    alpha=0.5,
    color='blue',
    label='Reference water'
)

ax.hist(
    amd_values,
    bins=bins,
    density=True,
    alpha=0.5,
    color='red',
    label='AMD water'
)


# OPTIONAL: MEAN LINES
ax.axvline(
    np.nanmean(water_values),
    color='darkblue',
    linestyle='--',
    linewidth=2
)

ax.axvline(
    np.nanmean(amd_values),
    color='darkred',
    linestyle='--',
    linewidth=2
)


# LABELS
ax.set_xlabel('(B4 - B2)', fontsize=11)

ax.set_ylabel('Density', fontsize=11)

ax.set_title(
    'Distribution of AMD spectral index values',
    fontsize=14,
    fontweight='bold'
)


# GRID
ax.grid(
    True,
    linestyle='--',
    alpha=0.3
)


# LEGEND
ax.legend()


# FINAL LAYOUT
plt.tight_layout()


# SAVE FIGURE
plt.savefig(
    hist_path,
    dpi=300,
    bbox_inches='tight'
)

print(f'Histogram saved to: {hist_path}')

plt.close()    
### 

### --- AMD / CLEAN WATER STATISTICS --- ###
json_path = os.path.join(
    output_dir,
    'amd_water_statistics.json'
)

# HELPER FUNCTION
def compute_statistics(values, variable_name, region_name):

    # Flatten
    values = values.flatten()

    # Raw pixel count
    raw_pixels = len(values)

    # Remove NaNs
    values = values[~np.isnan(values)]

    clear_pixels = len(values)

    # Outlier filtering (robust IQR)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = values[
        (values >= lower) &
        (values <= upper)
    ]

    outliers_removed = raw_pixels - len(filtered)

    # Mean
    mean = np.mean(filtered)

    # Median
    median = np.median(filtered)

    # Standard deviation
    std = np.std(filtered)

    # Median Absolute Deviation (MAD)
    mad = np.median(
        np.abs(filtered - median)
    )

    # Gaussian consistency scaling
    mad = mad * 1.4826


    # Statistics dictionary
    stats = {

    "variable": variable_name,

    "region": region_name,

    "observations": int(filtered.size),

    "mean_raw_pixels": float(raw_pixels),

    "total_outliers_removed": int(outliers_removed),

    "mean": float(mean),

    "median": float(median),

    "std": float(std),

    "mad": float(mad),

    "min": float(np.min(filtered)),

    "max": float(np.max(filtered)),

    "mean_clear_pixels": float(clear_pixels)
}

    return stats


# COMPUTE STATISTICS
stats_list = []

# AMD water statistics
stats_list.append(
    compute_statistics(
        peak_amd,
        variable_name='AMD_diff',
        region_name='amd_water'
    )
)

# Clean water statistics
stats_list.append(
    compute_statistics(
        peak_water,
        variable_name='AMD_diff',
        region_name='clean_water'
    )
)

# SAVE JSON
with open(json_path, 'w') as f:

    json.dump(
        stats_list,
        f,
        indent=2
    )

print(f'Statistics saved to: {json_path}')


# OPTIONAL: PRINT SUMMARY
for item in stats_list:

    print('\n--------------------------')

    print(f"Region: {item['region']}")

    print(f"Mean:   {item['mean']:.3f}")

    print(f"Median: {item['median']:.3f}")

    print(f"Std:    {item['std']:.3f}")

    print(f"MAD:    {item['mad']:.3f}")

    print(f"Min:    {item['min']:.3f}")

    print(f"Max:    {item['max']:.3f}")    



