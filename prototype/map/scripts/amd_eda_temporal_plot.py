"""The script:

1. Loads Sentinel-2 scenes
2. Uses AMD and water masks
3. Extracts pixels from AMD-affected lakes and reference lakes
4. Computes temporal mean values
5. Plots the temporal evolution for comparison
"""

import os
import glob
import numpy as np
import rasterio

from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import dates as matplotlib_dates



# INPUT DATA
# mac
# proj_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments' 
# skylake
proj_dir = '/home/lukas/ownCloud/Projects/GAIA_TSF/tsf_experiments/' 

# Directory with Sentinel-2 scenes
scenes_dir = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/inputs/sentinel2/') 

# predictions_dir = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/inputs/sentinel2_clouds/')

# AMD mask raster
amd_mask_path = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/static/yxsjoberg_binary_amd.tif')

# Water mask raster
water_mask_path = os.path.join(proj_dir, 'AMD_monitoring_Yxsjoberg/static/yxsjoberg_binary_water.tif')

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

# TEMPORAL AVERAGES
peak_amd = np.nanmean(peak_amd, axis=1)
peak_water = np.nanmean(peak_water, axis=1)

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
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)


# AMD lakes
ax.plot_date(
    date2num,
    peak_amd,
    'ro-',
    label='Lake affected by AMD'
)


# Reference lakes
ax.plot_date(
    date2num,
    peak_water,
    'bo-',
    label='Other lakes'
)


# Axis labels and title
ax.set_ylabel('(B4 - B2)')
plt.title(
    'Temporal change in (B4-B2) for lakes in Yxsjöberg area',
    fontsize=12
)


# Time range
ax.set_xlim([
    datetime(2018, 4, 15),
    datetime(2018, 10, 31)
])


# Formatting
ax.xaxis.set_major_locator(matplotlib_dates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(matplotlib_dates.DateFormatter('%d-%m-%Y'))

plt.legend()
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()
