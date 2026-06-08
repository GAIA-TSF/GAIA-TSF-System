
### GAIA-TSF ### 

# PURPOSE
# -------
# Generate EO-based AMD spectral indicators from
# Sentinel-2 imagery.
#
# WORKFLOW
# --------
# 1. Read Sentinel-2 scenes
# 2. Calculate cloud masks
# 3. Select cloud-free scenes
# 4. Calculate spectral indicators
# 5. Apply cloud masking
# 6. Save indicator rasters
#
# OUTPUT
# ------
# results/
#     amd_indicators/
#         amd_diff/
#             20180502T102031_amd_diff.tif
#         amd_ratio/
#             ...
#         amwi/
#             ...

import os
import glob
import yaml
import numpy as np
import pandas as pd
import rasterio

# SETTINGS 
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# INPUTS
inputs_dir = cfg["project"]["inputs_dir"]

tif_dir = os.path.join(
    inputs_dir,
    "sentinel2"
)

output_dir = os.path.join(
    cfg["project"]["output_dir"],
    "amd_indicators"
)

indicator_dir = output_dir

os.makedirs(
    indicator_dir,
    exist_ok=True
)


# CONFIGURATION

# AVAILABLE INDICATORS
AVAILABLE_INDICATORS = set(
    cfg["indices"]["available"]
)

ENABLED_INDICATORS = (
    cfg["indices"]["enabled"]
)


# CLOUD FILTERING
MAX_CLOUD_PERCENT = (
    cfg["clouds"]["max_cloud_percent"]
)

CLOUD_THRESHOLD = (
    cfg["clouds"]["threshold"]["blue_band_value"]
)

CLOUD_METHOD = (
    cfg["clouds"]["method"]
)

# VALIDATION
for indicator in ENABLED_INDICATORS:

    if indicator not in AVAILABLE_INDICATORS:

        raise ValueError(
            f"Unknown indicator: {indicator}"
        )


# FUNCTIONS
def calculate_cloud_mask(img, threshold=1000):

    """
    Simple threshold cloud masking.

    Sentinel-2:
        B2 = Blue band = index 1
    """

    blue = img[1]

    cloud_mask = (
        blue > threshold
    ).astype(np.uint8)

    return cloud_mask


def calculate_cloud_coverage(cloud_mask):

    """
    Percentage cloud coverage.
    """

    return 100 * np.mean(cloud_mask)

def calculate_scl_cloud_mask(
    img,
    cloud_classes
):

    scl = img[0]

    return np.isin(
        scl,
        cloud_classes
    ).astype(np.uint8)


# INDICATORS
def calculate_amd_diff(B2, B4):

    return B4 - B2


def calculate_amd_ratio(B2, B4):

    eps = 1e-6

    return B4 / (B2 + eps)


def calculate_amwi(B2, B4):

    eps = 1e-6

    return (
        (B4 + B2)
        /
        (B4 - B2 + eps)
    )


# ------------------------------------------------------------
# MAIN INDICATOR FUNCTION
# ------------------------------------------------------------

def calculate_amd_indices(img):

    """
    Sentinel-2 convention:
        B2 = Blue = index 1
        B4 = Red  = index 3
    """

    B2 = img[1].astype(np.float32)

    B4 = img[3].astype(np.float32)

    indices = {}

    if "AMD_diff" in ENABLED_INDICATORS:

        indices["AMD_diff"] = calculate_amd_diff(
            B2,
            B4
        )

    if "AMD_ratio" in ENABLED_INDICATORS:

        indices["AMD_ratio"] = calculate_amd_ratio(
            B2,
            B4
        )

    if "AMWI" in ENABLED_INDICATORS:

        indices["AMWI"] = calculate_amwi(
            B2,
            B4
        )

    return indices


# ------------------------------------------------------------
# SAVE RASTER
# ------------------------------------------------------------

def save_raster(output_path, array, reference_path):

    with rasterio.open(reference_path) as src:

        meta = src.meta.copy()

    meta.update({
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan
    })

    with rasterio.open(
        output_path,
        "w",
        **meta
    ) as dst:

        dst.write(
            array.astype(np.float32),
            1
        )

# ============================================================
# LOAD SCENES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            tif_dir,
            "*.tif"
        )
    )
)

print()
print(f"{len(files)} scenes found")

# ============================================================
# PROCESS SCENES
# ============================================================

data = []

for f in files:

    date_str = os.path.basename(f).split(".")[0]

    date = pd.to_datetime(date_str)

    print()
    print(date)

    with rasterio.open(f) as src:

        img = src.read()


    # CLOUD MASK
    if CLOUD_METHOD == "threshold":

        cloud_mask = calculate_cloud_mask(
            img,
            threshold=CLOUD_THRESHOLD
        )

    elif CLOUD_METHOD == "scl":

        f_scl = (
            f.replace(
                "sentinel2",
                "sentinel2_cloud"
            )
            .replace(
                ".tif",
                "_SCL.tif"
            )
        )

        with rasterio.open(f_scl) as src:

            scl = src.read()

        cloud_mask = calculate_scl_cloud_mask(
            scl,
            cfg["clouds"]["scl"]["classes"]
        )

    else:

        raise ValueError(
            f"Unknown cloud method: {CLOUD_METHOD}"
        )

    cloud_percent = calculate_cloud_coverage(
        cloud_mask
    )

    print(
        f"Cloud coverage: {cloud_percent:.2f}%"
    )

    # --------------------------------------------------------
    # CLOUD FILTERING
    # --------------------------------------------------------

    if cloud_percent > MAX_CLOUD_PERCENT:

        print("Scene discarded")

        continue

    print("Scene accepted")

    data.append({
        "date": date,
        "img": img,
        "cloud": cloud_mask,
        "path": f
    })

# ============================================================
# GENERATE INDICATORS
# ============================================================

print()
print("Generating spectral indicators...")

for d in data:

    date = d["date"]

    img = d["img"]

    cloud = d["cloud"]

    path = d["path"]

    date_str = date.strftime(
        "%Y%m%dT%H%M%S"
    )

    # --------------------------------------------------------
    # CALCULATE INDICES
    # --------------------------------------------------------

    indices = calculate_amd_indices(img)

    # --------------------------------------------------------
    # SAVE EACH INDICATOR
    # --------------------------------------------------------

    for index_name, arr in indices.items():

        print(
            f"Processing {date_str} | {index_name}"
        )

        # ----------------------------------------------------
        # APPLY CLOUD MASK
        # ----------------------------------------------------

        arr = np.where(
            cloud == 1,
            np.nan,
            arr
        )

        # ----------------------------------------------------
        # OUTPUT DIRECTORY
        # ----------------------------------------------------

        index_dir = os.path.join(
            indicator_dir,
            index_name.lower()
        )

        os.makedirs(
            index_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # OUTPUT FILE
        # ----------------------------------------------------

        output_path = os.path.join(
            index_dir,
            f"{date_str}_{index_name.lower()}.tif"
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_raster(
            output_path,
            arr,
            path
        )

        print(
            f"Saved: {output_path}"
        )


# DONE
print()
print("GAIA-TSF spectral indicator generation finished.")