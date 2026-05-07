"""
Exploratory Data Analysis for Sentinel-2 time series AMD monitoring data.

Prototype script:
- reads Sentinel-2 GeoTIFF time series
- applies separate cloud masks
- applies clean-water and TSF-water static masks
- computes descriptive statistics for each region
- displays histograms and temporal plots on screen only
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import glob
import re
import zlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
ASSUMED_BAND_NAMES = {
    1: "B1",
    2: "B2",
    3: "B3",
    4: "B4",
    5: "B5",
    6: "B6",
    7: "B7",
    8: "B8",
    9: "B8A",
    10: "B9",
    11: "B11",
    12: "B12",
}
EDA_BANDS = ("B2", "B3", "B4", "B8")


def parse_date_from_filename(path):
    """Parse dates like 20180716T102019.tif from a raster filename."""
    basename = os.path.basename(path)
    match = re.search(r"(\d{8})(?:T\d{6})?", basename)
    if not match:
        raise ValueError(f"Could not parse date from filename: {basename}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def find_cloud_mask(raster_path, cloud_dir):
    basename = os.path.basename(raster_path)
    stem, _ = os.path.splitext(basename)
    candidates = [
        os.path.join(cloud_dir, f"{stem}_pred.tif"),
        os.path.join(cloud_dir, basename),
        os.path.join(cloud_dir, f"{stem}.tif"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No cloud mask found for {basename}. Tried: {candidates}"
    )


def load_binary_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask > 0


def get_band_names(src):
    names = {}
    for idx in src.indexes:
        description = src.descriptions[idx - 1]
        if description:
            names[description] = idx
        elif idx in ASSUMED_BAND_NAMES:
            names[ASSUMED_BAND_NAMES[idx]] = idx
        else:
            names[f"band_{idx}"] = idx
    return names


def read_band_arrays(src, band_names):
    arrays = {}
    for band_name in EDA_BANDS:
        if band_name not in band_names:
            continue
        data = src.read(band_names[band_name]).astype("float32")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        arrays[band_name] = data

    if "B2" in arrays and "B4" in arrays:
        arrays["AMD_B4_B2"] = arrays["B4"] / (arrays["B2"] + 1e-6)

    return arrays


def calculate_stats(values):
    values = values[np.isfinite(values)]

    stats = {
        "count": int(values.size),
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "min": np.nan,
        "max": np.nan,
    }

    for percentile in PERCENTILES:
        stats[f"p{percentile}"] = np.nan

    if values.size == 0:
        return stats

    stats.update(
        {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
    )
    stats.update(
        {
            f"p{percentile}": float(np.nanpercentile(values, percentile))
            for percentile in PERCENTILES
        }
    )
    return stats


def sample_for_histogram(values, max_samples=3000, seed=42):
    values = values[np.isfinite(values)]
    if values.size <= max_samples:
        return values

    rng = np.random.default_rng(seed)
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx]


def summarize_raster(
    raster_path,
    cloud_dir,
    clean_mask,
    tsf_mask,
    histogram_values,
):
    date = parse_date_from_filename(raster_path)
    cloud_path = find_cloud_mask(raster_path, cloud_dir)

    with rasterio.open(raster_path) as src:
        band_names = get_band_names(src)
        arrays = read_band_arrays(src, band_names)

    with rasterio.open(cloud_path) as src:
        cloud = src.read(1)

    reference_shape = next(iter(arrays.values())).shape if arrays else None
    for name, candidate in (
        ("cloud mask", cloud),
        ("clean-water mask", clean_mask),
        ("TSF-water mask", tsf_mask),
    ):
        if reference_shape is not None and candidate.shape != reference_shape:
            raise ValueError(
                f"Shape mismatch for {name} while processing "
                f"{os.path.basename(raster_path)}: expected {reference_shape}, "
                f"got {candidate.shape}"
            )

    clear_sky = cloud == 0
    rows = []
    regions = {
        "clean_water": clean_mask & clear_sky,
        "tsf_water": tsf_mask & clear_sky,
    }

    for variable, array in arrays.items():
        for region_name, region_mask in regions.items():
            values = array[region_mask]
            row = {
                "date": date,
                "raster": os.path.basename(raster_path),
                "cloud_mask": os.path.basename(cloud_path),
                "region": region_name,
                "variable": variable,
            }
            row.update(calculate_stats(values))
            rows.append(row)

            sample = sample_for_histogram(
                values,
                seed=zlib.crc32(f"{date}-{region_name}-{variable}".encode("utf-8")),
            )
            histogram_values[(variable, region_name)].append(sample)

    return rows


def build_eda_table(tif_dir, cloud_dir, clean_mask_path, tsf_mask_path):
    raster_paths = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
    if not raster_paths:
        raise FileNotFoundError(f"No Sentinel-2 GeoTIFFs found in: {tif_dir}")

    clean_mask = load_binary_mask(clean_mask_path)
    tsf_mask = load_binary_mask(tsf_mask_path)
    histogram_values = defaultdict(list)
    all_rows = []

    print(f"[INFO] Found {len(raster_paths)} Sentinel-2 rasters")
    print(f"[INFO] Clean-water mask pixels: {int(clean_mask.sum())}")
    print(f"[INFO] TSF-water mask pixels: {int(tsf_mask.sum())}")

    for i, raster_path in enumerate(raster_paths, start=1):
        print(f"[INFO] Processing {i}/{len(raster_paths)}: {os.path.basename(raster_path)}")
        rows = summarize_raster(
            raster_path,
            cloud_dir,
            clean_mask,
            tsf_mask,
            histogram_values,
        )
        all_rows.extend(rows)

    stats_df = pd.DataFrame(all_rows).sort_values(["date", "variable", "region"])
    return stats_df, histogram_values


def print_results(stats_df):
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)

    print("\n=== Per-Date Statistics ===")
    print(stats_df.to_string(index=False))

    print("\n=== Overall Summary Across Dates ===")
    summary = (
        stats_df.groupby(["variable", "region"])
        .agg(
            observations=("date", "count"),
            mean_of_means=("mean", "mean"),
            median_of_medians=("median", "median"),
            mean_std=("std", "mean"),
            min_observed=("min", "min"),
            max_observed=("max", "max"),
            mean_clear_pixels=("count", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


def plot_histograms(histogram_values):
    variables = sorted({key[0] for key in histogram_values})

    for variable in variables:
        plt.figure(figsize=(10, 5))
        for region_name, color in (
            ("clean_water", "tab:blue"),
            ("tsf_water", "tab:orange"),
        ):
            chunks = histogram_values.get((variable, region_name), [])
            if not chunks:
                continue
            values = np.concatenate(chunks)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            plt.hist(
                values,
                bins=60,
                alpha=0.45,
                density=True,
                color=color,
                label=f"{region_name} (n={values.size})",
            )

        plt.title(f"Histogram: {variable}")
        plt.xlabel(variable)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_temporal_statistics(stats_df):
    variables = sorted(stats_df["variable"].unique())

    for variable in variables:
        df = stats_df[stats_df["variable"] == variable]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for region_name, color in (
            ("clean_water", "tab:blue"),
            ("tsf_water", "tab:orange"),
        ):
            region_df = df[df["region"] == region_name].sort_values("date")
            if region_df.empty:
                continue

            axes[0].plot(
                region_df["date"],
                region_df["mean"],
                marker="o",
                linewidth=1.5,
                color=color,
                label=f"{region_name} mean",
            )
            axes[0].plot(
                region_df["date"],
                region_df["median"],
                marker=".",
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.75,
                label=f"{region_name} median",
            )

            axes[1].plot(
                region_df["date"],
                region_df["p25"],
                linestyle=":",
                linewidth=1.0,
                color=color,
                alpha=0.8,
                label=f"{region_name} p25",
            )
            axes[1].plot(
                region_df["date"],
                region_df["p75"],
                linestyle="-",
                linewidth=1.0,
                color=color,
                alpha=0.8,
                label=f"{region_name} p75",
            )
            axes[1].fill_between(
                region_df["date"],
                region_df["p25"],
                region_df["p75"],
                color=color,
                alpha=0.15,
            )

        axes[0].set_title(f"Temporal Mean and Median: {variable}")
        axes[0].set_ylabel(variable)
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        axes[1].set_title(f"Interquartile Range: {variable}")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel(variable)
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.25)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()


def main():
    inputs_dir = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/inputs/"
    tif_dir = os.path.join(inputs_dir, "sentinel2")
    cloud_dir = os.path.join(inputs_dir, "sentinel2_clouds")

    static_dir = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/static/"
    clean_mask_path = os.path.join(static_dir, "yxsjoberg_clean_water_mask.tif")
    tsf_mask_path = os.path.join(static_dir, "yxsjoberg_tsf_water_mask.tif")

    output_dir = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/results/monitoring"

    print("[INFO] Sentinel-2 AMD monitoring EDA")
    print(f"[INFO] Raster directory: {tif_dir}")
    print(f"[INFO] Cloud-mask directory: {cloud_dir}")
    print(f"[INFO] Clean-water mask: {clean_mask_path}")
    print(f"[INFO] TSF-water mask: {tsf_mask_path}")
    print(f"[INFO] Output directory configured but unused for this screen-only EDA: {output_dir}")

    stats_df, histogram_values = build_eda_table(
        tif_dir=tif_dir,
        cloud_dir=cloud_dir,
        clean_mask_path=clean_mask_path,
        tsf_mask_path=tsf_mask_path,
    )

    print_results(stats_df)
    plot_histograms(histogram_values)
    plot_temporal_statistics(stats_df)


if __name__ == "__main__":
    main()
