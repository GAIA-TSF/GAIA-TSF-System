### GAIA-TSF ###
#
# Generate temporal dynamics features from
# EO-derived AMD spectral indicators.
#
# INPUT
# -----
# results/
#     amd_indicators/
#         amd_diff/
#         amd_ratio/
#         amwi/
#     temporal_features/
#         amd_diff/
#             smooth/
#             lag1/
#             lag2/
#             lag3/
#             diff1/
#             diff2/
#             acc/
#             roll_mean/
#             roll_std/
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

from scipy.signal import savgol_filter


# SETTINGS 
# with open("/media/lukas/image2/GAIA_TSF/src/GAIA-TSF-System/prototype/map/scripts/config.yaml") as f:
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
 

# OUTPUTS 
results_dir = cfg["project"]["output_dir"]

indicator_dir = os.path.join(
    results_dir,
    "amd_indicators"
)

temporal_dir = os.path.join(
    results_dir,
    "temporal_features"
)

os.makedirs(
    temporal_dir,
    exist_ok=True
)

# ENABLED INDICATORS
ENABLED_INDICATORS = (
    cfg["indices"]["enabled"]
)

# TEMPORAL FEATURE REGISTRY
AVAILABLE_TEMPORAL_FEATURES = (
    cfg["temporal_features"]
)

ENABLED_TEMPORAL_FEATURES = [

    name

    for name, params in
    AVAILABLE_TEMPORAL_FEATURES.items()

    if params.get("enabled", False)
]


# PARAMETERS
SAVGOL_WINDOW = (
    cfg["smoothing"]["window_length"]
)

SAVGOL_POLYORDER = (
    cfg["smoothing"]["polyorder"]
)


# VALIDATION

for feature in ENABLED_TEMPORAL_FEATURES:

    if feature not in AVAILABLE_TEMPORAL_FEATURES:

        raise ValueError(
            f"Unknown temporal feature: {feature}"
        )


# FUNCTIONS
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


# TEMPORAL FUNCTIONS
def smooth_signal(ts):

    """
    Savitzky-Golay smoothing.
    """

    valid = np.isfinite(ts)

    if np.sum(valid) < SAVGOL_WINDOW:

        return ts

    ts_interp = (
        pd.Series(ts)
        .interpolate(limit_direction='both')
        .values
    )

    smooth = savgol_filter(
        ts_interp,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER
    )

    smooth[~valid] = np.nan

    return smooth


def temporal_lag(ts, lag=1):

    out = np.full_like(
        ts,
        np.nan,
        dtype=np.float32
    )

    out[lag:] = ts[:-lag]

    return out


def temporal_diff(ts, order=1):

    out = ts.copy()

    for _ in range(order):

        out = np.diff(
            out,
            prepend=np.nan
        )

    return out


def temporal_acc(ts):

    d1 = temporal_diff(
        ts,
        order=1
    )

    return np.gradient(d1)


def temporal_rolling(ts, window=5, mode="mean"):

    s = pd.Series(ts)

    if mode == "mean":

        return (
            s.rolling(
                window,
                center=True,
                min_periods=1
            )
            .mean()
            .values
        )

    elif mode == "std":

        return (
            s.rolling(
                window,
                center=True,
                min_periods=1
            )
            .std()
            .values
        )


# PROCESS INDICATORS
print()
print("Generating temporal features...")

for indicator_name in ENABLED_INDICATORS:

    print()
    print(f"Indicator: {indicator_name}")


    # INPUT FILES
    input_dir = os.path.join(
        indicator_dir,
        indicator_name.lower()
    )

    files = sorted(
        glob.glob(
            os.path.join(
                input_dir,
                "*.tif"
            )
        )
    )

    print(f"{len(files)} scenes")

    if len(files) == 0:

        print("No files found")

        continue

    # --------------------------------------------------------
    # LOAD STACK
    # --------------------------------------------------------

    stack = []

    for f in files:

        with rasterio.open(f) as src:

            arr = src.read(1)

        stack.append(arr)

    stack = np.stack(stack)

    # shape:
    # (T, H, W)

    T, H, W = stack.shape

    print(stack.shape)

    # --------------------------------------------------------
    # FEATURE STACKS
    # --------------------------------------------------------

    feature_stacks = {}

    for feature_name in ENABLED_TEMPORAL_FEATURES:
        # print(f"Initializing stack for feature: {feature_name}") 

        feature_stacks[feature_name] = np.full(
            stack.shape,
            np.nan,
            dtype=np.float32
        )


    # PIXEL-WISE TEMPORAL ANALYSIS
    for y in range(H):

        if y % 50 == 0:

            print(f"Row {y}/{H}")

        for x in range(W):

            ts = stack[:, y, x]

            # SKIP EMPTY PIXELS
            if np.all(np.isnan(ts)):

                continue

            # TEMPORAL INTERPOLATION
            ts_interp = (
                pd.Series(ts)
                .interpolate(limit_direction='both')
                .values
            )

            # BASE SMOOTHED SIGNAL
            smooth_base = smooth_signal(
                ts_interp
            )

            # optionally store smooth signal
            if "smooth" in ENABLED_TEMPORAL_FEATURES:

                feature_stacks["smooth"][:, y, x] = smooth_base


            # GENERATE FEATURES
            for feature_name in ENABLED_TEMPORAL_FEATURES:

                # already handled
                if feature_name == "smooth":

                    continue

                config = AVAILABLE_TEMPORAL_FEATURES[
                    feature_name
                ]

                feature_type = config["type"]

                # LAG FEATURES
                if feature_type == "lag":

                    feature = temporal_lag(
                        smooth_base,
                        lag=config["lag"]
                    )

                # DIFFERENCES
                elif feature_type == "diff":

                    feature = temporal_diff(
                        smooth_base,
                        order=config["order"]
                    )

                # ACCELERATION
                elif feature_type == "acc":

                    feature = temporal_acc(
                        smooth_base
                    )

                # ROLLING MEAN
                elif feature_type == "rolling_mean":

                    feature = temporal_rolling(
                        smooth_base,
                        window=config["window"],
                        mode="mean"
                    )

                # ROLLING STD
                elif feature_type == "rolling_std":

                    feature = temporal_rolling(
                        smooth_base,
                        window=config["window"],
                        mode="std"
                    )

                else:

                    continue

                # STORE
                feature_stacks[feature_name][:, y, x] = feature

    
    # SAVE FEATURES
    
    print()
    print("Saving features...")

    for feature_name, feature_stack in feature_stacks.items():

        print(feature_name)

        feature_dir = os.path.join(
            temporal_dir,
            indicator_name,
            feature_name
        )

        os.makedirs(
            feature_dir,
            exist_ok=True
        )

        for i, f in enumerate(files):

            basename = os.path.basename(f)

            output_path = os.path.join(
                feature_dir,
                basename.replace(
                    indicator_name,
                    feature_name
                )
            )

            save_raster(
                output_path,
                feature_stack[i],
                f
            )


# DONE
print()
print("GAIA-TSF temporal feature generation finished.")

