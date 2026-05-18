"""
Exploratory Data Analysis for Sentinel-2 time series AMD monitoring data.

Prototype script:
- reads Sentinel-2 GeoTIFF time series
- calculate and applies separate cloud masks
- applies clean-water and AMD-water static masks
- detects and removes outliers for clean-water and AMD-water AMD_ratio and AMD_diff
- saves histograms, temporal plots, and overall JSON summary
"""

import os
import glob
import json
import numpy as np
import rasterio

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import dates as matplotlib_dates



# INPUT DATA
# mac
# proj_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments' 
# skylake
proj_dir = '/home/lukas/ownCloud/Projects/GAIA_TSF/tsf_experiments/' 

# RESULTS DIRECTORY
res_dir = os.path.join(
    proj_dir,
    'AMD_monitoring_Yxsjoberg/results/eda'
)

# Create directory if it does not exist
os.makedirs(res_dir, exist_ok=True)


# Directory with Sentinel-2 scenes
scenes_dir = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/inputs/sentinel2/') 

# predictions_dir = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/inputs/sentinel2_clouds/')

# AMD mask raster
amd_mask_path = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/static/yxsjoberg_binary_amd.tif')

# Water mask raster
water_mask_path = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/static/yxsjoberg_clean_water_mask.tif')

# CONSTATNTS 
codes = [1, 2, 3, 4, 5, 6]

code_names = [
    'Cloud',
    'AMD',
    'Water',
    'Bare Soils',
    'Grassland',
    'Woodland'
]

CLOUD_CODE = 1

# LOAD INPUTS
# List all Sentinel-2 scenes
# cloudless_scenes = sorted(glob.glob(os.path.join(scenes_dir, "*.tif")))
# print(f"{len(cloudless_scenes)} scenes found")

# COMPUTE CLOUD COVER
# Sentinel-2 scenes
fileList = sorted(glob.glob(os.path.join(scenes_dir, "*.tif")))
print(f"{len(fileList)} Sentinel-2 scenes found")

# Store cloud cover percentages
cloud_cover = []

# Iterate through scenes
for i, path in enumerate(fileList):

    filename = os.path.basename(path)
    scene_id = filename.replace(".tif", "")

    print(f"Processing {scene_id} ({i+1}/{len(fileList)})")

    # Open Sentinel-2 scene
    with rasterio.open(path) as raster:

        # Cloud mask from Band 2 threshold
        # Pixels < 1000 are considered clouds
        cloud_mask = (raster.read(2) < 1000).astype("uint8")

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
cloud_threshold = 10.0
indices = np.where(np.array(cloud_cover) < cloud_threshold)[0]
cloudless_scenes = [fileList[i] for i in indices]
print(len(cloudless_scenes), 'scenes were kept')
print(len(fileList) - len(cloudless_scenes), 'scenes were discarded')    
    

# Open AMD mask
with rasterio.open(amd_mask_path) as src:
    binary_amd = src.read()

# Open water mask
with rasterio.open(water_mask_path) as src:
    binary_water = src.read()


# GET PIXELs
ay, ax = np.where(binary_amd[0, :, :] == 1)

# Reference lake pixels
wy, wx = np.where(binary_water[0, :, :] == 1)

# CREATE ARRAYS FOR TEMPORAL VALUES
peak_amd = np.zeros((len(cloudless_scenes), len(ax)))
peak_water = np.zeros((len(cloudless_scenes), len(wx)))

valid_amd_pixels = []
valid_water_pixels = []

# PROCESS EACH SCENE
for i, path in enumerate(cloudless_scenes):
    # break 

    print(f"Processing scene {i+1}/{len(cloudless_scenes)}")

    filename = os.path.basename(path)
    scene_id = filename.replace(".tif", "")


    # LOAD DATA
    with rasterio.open(path, 'r') as raster:

        # Compute spectral difference
        diff = raster.read(4).astype(np.float32) - raster.read(2).astype(np.float32)


    # track number of valid pixels 
    amd_values = diff[ay, ax]
    water_values = diff[wy, wx]

    peak_amd[i, :] = amd_values
    peak_water[i, :] = water_values

    valid_amd_pixels.append(np.sum(~np.isnan(amd_values)))
    valid_water_pixels.append(np.sum(~np.isnan(water_values)))

# TEMPORAL AVERAGES XXXX
# peak_amd = np.nanmean(peak_amd, axis=1)
# peak_water = np.nanmean(peak_water, axis=1)

# EXTRACT DATES FROM FILENAMES
# Expected filename example: 20180615T103021.tif
dates = []

for path in cloudless_scenes:

    filename = os.path.basename(path)

    time = filename.replace('.tif', '')
    time = time.split('T')[0]

    datetime_object = datetime.strptime(time, '%Y%m%d')

    dates.append(datetime_object)


# Convert dates to matplotlib format

date2num = matplotlib_dates.date2num(dates)


### --- PLOT RESULTS --- ### 


# OUTPUT FIGURE PATH
fig_path = os.path.join(
    res_dir,
    'temporal_amd_profiles.png'
)

# CREATE FIGURE
fig, ax = plt.subplots(figsize=(12, 6))

# ------------------------------------
# PLOT ALL AMD PIXEL TEMPORAL PROFILES
# ------------------------------------
for j in range(peak_amd.shape[1]):

    ax.plot(
        dates,
        peak_amd[:, j],
        color='red',
        alpha=0.05,
        linewidth=0.6
    )

# ----------------------------------------
# PLOT ALL REFERENCE WATERBODY PROFILES
# ----------------------------------------
for j in range(peak_water.shape[1]):

    ax.plot(
        dates,
        peak_water[:, j],
        color='blue',
        alpha=0.02,
        linewidth=0.5
    )

# ------------------------------------
# OVERLAY TEMPORAL MEANS
# ------------------------------------
amd_mean = np.nanmean(peak_amd, axis=1)
water_mean = np.nanmean(peak_water, axis=1)

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

# ------------------------------------
# OPTIONAL: PERCENTILE ENVELOPES
# ------------------------------------
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

# ------------------------------------
# LABELS
# ------------------------------------
ax.set_ylabel('(B4 - B2)', fontsize=11)

ax.set_xlabel('Date', fontsize=11)

ax.set_title(
    'Temporal AMD spectral dynamics',
    fontsize=14,
    fontweight='bold'
)

# ------------------------------------
# GRID
# ------------------------------------
ax.grid(
    True,
    linestyle='--',
    alpha=0.3
)

# ------------------------------------
# TIME RANGE
# ------------------------------------
ax.set_xlim([
    datetime(2018, 4, 15),
    datetime(2018, 10, 31)
])

# ------------------------------------
# DATE FORMATTING
# ------------------------------------
ax.xaxis.set_major_locator(
    matplotlib_dates.MonthLocator(interval=1)
)

ax.xaxis.set_major_formatter(
    matplotlib_dates.DateFormatter('%d-%m-%Y')
)

fig.autofmt_xdate()

# ------------------------------------
# LEGEND
# ------------------------------------
ax.legend(
    loc='upper right',
    frameon=True
)

# ------------------------------------
# FINAL LAYOUT
# ------------------------------------
plt.tight_layout()

# ------------------------------------
# SAVE FIGURE
# ------------------------------------
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
    res_dir,
    'histogram_amd_vs_clean_water.png'
)

# ------------------------------------
# FLATTEN ARRAYS
# ------------------------------------
amd_values = peak_amd.flatten()
water_values = peak_water.flatten()

# REMOVE NaNs
amd_values = amd_values[~np.isnan(amd_values)]
water_values = water_values[~np.isnan(water_values)]

# ------------------------------------
# CREATE FIGURE
# ------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# ------------------------------------
# HISTOGRAMS
# ------------------------------------
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

# ------------------------------------
# OPTIONAL: MEAN LINES
# ------------------------------------
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

# ------------------------------------
# LABELS
# ------------------------------------
ax.set_xlabel('(B4 - B2)', fontsize=11)

ax.set_ylabel('Density', fontsize=11)

ax.set_title(
    'Distribution of AMD spectral index values',
    fontsize=14,
    fontweight='bold'
)

# ------------------------------------
# GRID
# ------------------------------------
ax.grid(
    True,
    linestyle='--',
    alpha=0.3
)

# ------------------------------------
# LEGEND
# ------------------------------------
ax.legend()

# ------------------------------------
# FINAL LAYOUT
# ------------------------------------
plt.tight_layout()

# ------------------------------------
# SAVE FIGURE
# ------------------------------------
plt.savefig(
    hist_path,
    dpi=300,
    bbox_inches='tight'
)

print(f'Histogram saved to: {hist_path}')

plt.close()




### --- AMD / CLEAN WATER STATISTICS --- ###

# OUTPUT JSON PATH
json_path = os.path.join(
    res_dir,
    'amd_water_statistics.json'
)

# ------------------------------------
# HELPER FUNCTION
# ------------------------------------
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

# ------------------------------------
# COMPUTE STATISTICS
# ------------------------------------
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

# ------------------------------------
# SAVE JSON
# ------------------------------------
with open(json_path, 'w') as f:

    json.dump(
        stats_list,
        f,
        indent=2
    )

print(f'Statistics saved to: {json_path}')

# ------------------------------------
# OPTIONAL: PRINT SUMMARY
# ------------------------------------
for item in stats_list:

    print('\n--------------------------')

    print(f"Region: {item['region']}")

    print(f"Mean:   {item['mean']:.3f}")

    print(f"Median: {item['median']:.3f}")

    print(f"Std:    {item['std']:.3f}")

    print(f"MAD:    {item['mad']:.3f}")

    print(f"Min:    {item['min']:.3f}")

    print(f"Max:    {item['max']:.3f}") 

