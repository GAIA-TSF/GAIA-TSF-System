"""
Exploratory Data Analysis for Sentinel-2 time series AMD monitoring data.

Prototype script:
- reads Sentinel-2 GeoTIFF time series
- applies separate cloud masks
- applies clean-water and AMD-water static masks
- detects and removes outliers for clean-water and AMD-water AMD_ratio and AMD_diff
- saves histograms, temporal plots, and overall JSON summary
"""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import glob
import re
import zlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


PERCENTILES = (5, 50, 95)
TEMPORAL_SAMPLE_SIZE = 10
RANDOM_SEED = 42
OUTLIER_MAD_Z_THRESHOLD = 3.5
AMD_RATIO_MAX = 20
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
REQUIRED_BANDS = ("B2", "B4")


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


def read_band(src, band_names, band_name):
    if band_name not in band_names:
        raise KeyError(
            f"Required band {band_name} not found. Available bands: {sorted(band_names)}"
        )

    data = src.read(band_names[band_name]).astype("float32")
    if src.nodata is not None:
        data[data == src.nodata] = np.nan
    return data


def calculate_amd_arrays(src, band_names):
    for band_name in REQUIRED_BANDS:
        if band_name not in band_names:
            raise KeyError(
                f"Required band {band_name} not found. Available bands: {sorted(band_names)}"
            )

    b2 = read_band(src, band_names, "B2")
    b4 = read_band(src, band_names, "B4")

    with np.errstate(divide="ignore", invalid="ignore"):
        amd_ratio = b4 / (b2 + 1e-6)

    return {
        "AMD_ratio": amd_ratio,
        "AMD_diff": b4 - b2,
    }


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


def detect_outliers(values, z_threshold=OUTLIER_MAD_Z_THRESHOLD):
    values = np.asarray(values, dtype="float32")
    finite = np.isfinite(values)
    outliers = np.zeros(values.shape, dtype=bool)
    finite_values = values[finite]

    if finite_values.size == 0:
        return outliers

    median = np.nanmedian(finite_values)
    mad = np.nanmedian(np.abs(finite_values - median))

    if mad == 0 or not np.isfinite(mad):
        q1, q3 = np.nanpercentile(finite_values, [25, 75])
        iqr = q3 - q1
        if iqr == 0 or not np.isfinite(iqr):
            return outliers
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[finite] = (finite_values < lower) | (finite_values > upper)
        return outliers

    robust_z = 0.6745 * (finite_values - median) / mad
    outliers[finite] = np.abs(robust_z) > z_threshold
    return outliers


def detect_variable_outliers(values, variable):
    outliers = detect_outliers(values)
    if variable == "AMD_ratio":
        outliers |= np.asarray(values) > AMD_RATIO_MAX
    return outliers


def detect_outliers_against_reference(
    values,
    reference_values,
    z_threshold=OUTLIER_MAD_Z_THRESHOLD,
):
    values = np.asarray(values, dtype="float32")
    reference_values = np.asarray(reference_values, dtype="float32")
    finite_values = np.isfinite(values)
    finite_reference = np.isfinite(reference_values)
    outliers = np.zeros(values.shape, dtype=bool)
    reference_values = reference_values[finite_reference]

    if reference_values.size == 0:
        return outliers

    median = np.nanmedian(reference_values)
    mad = np.nanmedian(np.abs(reference_values - median))

    if mad == 0 or not np.isfinite(mad):
        q1, q3 = np.nanpercentile(reference_values, [25, 75])
        iqr = q3 - q1
        if iqr == 0 or not np.isfinite(iqr):
            return outliers
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[finite_values] = (
            (values[finite_values] < lower) | (values[finite_values] > upper)
        )
        return outliers

    robust_z = 0.6745 * (values[finite_values] - median) / mad
    outliers[finite_values] = np.abs(robust_z) > z_threshold
    return outliers


def detect_variable_outliers_against_reference(values, reference_values, variable):
    outliers = detect_outliers_against_reference(values, reference_values)
    if variable == "AMD_ratio":
        outliers |= np.asarray(values) > AMD_RATIO_MAX
    return outliers


def sample_for_histogram(values, max_samples=3000, seed=42):
    values = values[np.isfinite(values)]
    if values.size <= max_samples:
        return values

    rng = np.random.default_rng(seed)
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx]


def sample_mask_pixels(mask, n=TEMPORAL_SAMPLE_SIZE, seed=RANDOM_SEED):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return ys[idx], xs[idx]


def summarize_raster(
    raster_path,
    cloud_dir,
    clean_mask,
    tsf_mask,
    histogram_values,
    temporal_values,
    temporal_sample_pixels,
):
    date = parse_date_from_filename(raster_path)
    cloud_path = find_cloud_mask(raster_path, cloud_dir)

    with rasterio.open(raster_path) as src:
        band_names = get_band_names(src)
        arrays = calculate_amd_arrays(src, band_names)

    with rasterio.open(cloud_path) as src:
        cloud = src.read(1)

    reference_shape = next(iter(arrays.values())).shape if arrays else None
    for name, candidate in (
        ("cloud mask", cloud),
        ("clean-water mask", clean_mask),
        ("AMD-water mask", tsf_mask),
    ):
        if reference_shape is not None and candidate.shape != reference_shape:
            raise ValueError(
                f"Shape mismatch for {name} while processing "
                f"{os.path.basename(raster_path)}: expected {reference_shape}, "
                f"got {candidate.shape}"
            )

    # Cloud classifier convention: 1 = cloud, all other values are valid.
    clear_sky = cloud != 1
    rows = []
    regions = {
        "clean_water": clean_mask & clear_sky,
        "amd_water": tsf_mask & clear_sky,
    }

    for variable, array in arrays.items():
        for region_name, region_mask in regions.items():
            values = array[region_mask]
            values = values[np.isfinite(values)]
            outlier_mask = detect_variable_outliers(values, variable)
            clean_values = values[~outlier_mask]

            row = {
                "date": date,
                "raster": os.path.basename(raster_path),
                "cloud_mask": os.path.basename(cloud_path),
                "region": region_name,
                "variable": variable,
                "raw_count": int(values.size),
                "outliers_removed": int(outlier_mask.sum()),
            }
            row.update(calculate_stats(clean_values))
            rows.append(row)

            sample = sample_for_histogram(
                clean_values,
                seed=zlib.crc32(f"{date}-{region_name}-{variable}".encode("utf-8")),
            )
            histogram_values[(variable, region_name)].append(sample)

            sample_ys, sample_xs = temporal_sample_pixels[region_name]
            sample_values = array[sample_ys, sample_xs]
            sample_clear = clear_sky[sample_ys, sample_xs]
            sample_finite = np.isfinite(sample_values)
            sample_valid = sample_clear & sample_finite

            sample_outliers = detect_variable_outliers_against_reference(
                sample_values,
                values,
                variable,
            )

            temporal_values[(variable, region_name)].append(
                {
                    "date": date,
                    "clean_values": sample_values[sample_valid & ~sample_outliers],
                    "outlier_values": sample_values[sample_valid & sample_outliers],
                }
            )

    return rows


def build_eda_table(tif_dir, cloud_dir, clean_mask_path, tsf_mask_path):
    raster_paths = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
    if not raster_paths:
        raise FileNotFoundError(f"No Sentinel-2 GeoTIFFs found in: {tif_dir}")

    clean_mask = load_binary_mask(clean_mask_path)
    tsf_mask = load_binary_mask(tsf_mask_path)
    temporal_sample_pixels = {
        "clean_water": sample_mask_pixels(clean_mask, seed=RANDOM_SEED),
        "amd_water": sample_mask_pixels(tsf_mask, seed=RANDOM_SEED + 1),
    }
    histogram_values = defaultdict(list)
    temporal_values = defaultdict(list)
    all_rows = []

    print(f"[INFO] Found {len(raster_paths)} Sentinel-2 rasters")
    print(f"[INFO] Clean-water mask pixels: {int(clean_mask.sum())}")
    print(f"[INFO] AMD-water mask pixels: {int(tsf_mask.sum())}")
    print(f"[INFO] Clean-water temporal sample pixels: {list(zip(*temporal_sample_pixels['clean_water']))}")
    print(f"[INFO] AMD-water temporal sample pixels: {list(zip(*temporal_sample_pixels['amd_water']))}")

    for i, raster_path in enumerate(raster_paths, start=1):
        print(f"[INFO] Processing {i}/{len(raster_paths)}: {os.path.basename(raster_path)}")
        rows = summarize_raster(
            raster_path,
            cloud_dir,
            clean_mask,
            tsf_mask,
            histogram_values,
            temporal_values,
            temporal_sample_pixels,
        )
        all_rows.extend(rows)

    stats_df = pd.DataFrame(all_rows).sort_values(["date", "variable", "region"])
    return stats_df, histogram_values, temporal_values


def make_overall_summary(stats_df):
    return (
        stats_df.groupby(["variable", "region"])
        .agg(
            observations=("date", "count"),
            mean_raw_pixels=("raw_count", "mean"),
            total_outliers_removed=("outliers_removed", "sum"),
            mean_of_means=("mean", "mean"),
            median_of_medians=("median", "median"),
            mean_std=("std", "mean"),
            min_observed=("min", "min"),
            max_observed=("max", "max"),
            mean_clear_pixels=("count", "mean"),
        )
        .reset_index()
    )


def print_results(stats_df, summary_df):
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)

    print("\n=== Per-Date Statistics ===")
    print(stats_df.to_string(index=False))

    print("\n=== Overall Summary Across Dates ===")
    print(summary_df.to_string(index=False))


def save_overall_summary_json(summary_df, output_dir):
    output_path = os.path.join(output_dir, "overall_summary_across_dates.json")
    summary_df.to_json(output_path, orient="records", indent=2)
    print(f"[INFO] Saved overall summary JSON: {output_path}")


def add_legend_if_needed(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(loc="best")


def plot_histograms(histogram_values, output_dir):
    variables = sorted({key[0] for key in histogram_values})

    for variable in variables:
        plt.figure(figsize=(10, 5))
        for region_name, color in (
            ("clean_water", "tab:blue"),
            ("amd_water", "tab:orange"),
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
                label=f"{region_name} cleaned (n={values.size})",
            )

        plt.title(f"Histogram After Outlier Removal: {variable}")
        plt.xlabel(variable)
        plt.ylabel("Density")
        add_legend_if_needed(plt.gca())
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"histogram_{variable}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved histogram: {output_path}")


def plot_temporal_statistics(stats_df, temporal_values, output_dir):
    variables = sorted(stats_df["variable"].unique())

    for variable in variables:
        df = stats_df[stats_df["variable"] == variable]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        cleaned_plot_values = []
        outlier_dates_by_region = defaultdict(list)
        plot_series = []

        for region_name, color in (
            ("clean_water", "tab:blue"),
            ("amd_water", "tab:orange"),
        ):
            region_df = df[df["region"] == region_name].sort_values("date")
            if region_df.empty:
                continue
            if region_df[["mean", "std"]].isna().all().all():
                continue

            plot_df = region_df.set_index("date")[["mean", "std"]]
            plot_df = plot_df.interpolate(method="time", limit_direction="both")
            plot_df = plot_df.reset_index()
            lower_std = plot_df["mean"] - plot_df["std"]
            upper_std = plot_df["mean"] + plot_df["std"]
            plot_series.append((region_name, color, plot_df, lower_std, upper_std))

            for item in temporal_values.get((variable, region_name), []):
                date = item["date"]
                clean_values = item["clean_values"]
                outlier_values = item["outlier_values"]

                if clean_values.size > 0:
                    cleaned_plot_values.append(clean_values)
                    axes[0].scatter(
                        np.repeat(date, clean_values.size),
                        clean_values,
                        s=18,
                        alpha=0.5,
                        color=color,
                        edgecolors="none",
                        label=f"{region_name} sampled points",
                    )

                if outlier_values.size > 0:
                    outlier_dates_by_region[region_name].append(date)

        axis_values = []
        for _, _, plot_df, lower_std, upper_std in plot_series:
            axis_values.extend([
                plot_df["mean"].to_numpy(),
                lower_std.to_numpy(),
                upper_std.to_numpy(),
            ])
        if cleaned_plot_values:
            axis_values.append(np.concatenate(cleaned_plot_values))

        if axis_values:
            axis_values = np.concatenate(axis_values)
            axis_values = axis_values[np.isfinite(axis_values)]
        else:
            axis_values = np.array([])

        if axis_values.size > 0:
            y_min = float(np.nanmin(axis_values))
            y_max = float(np.nanmax(axis_values))
            y_range = y_max - y_min
            padding = y_range * 0.08 if y_range > 0 else max(abs(y_min) * 0.08, 1.0)
            y_lower = y_min - padding
            y_upper = y_max + padding
            axes[0].set_ylim(y_lower, y_upper)
            axes[1].set_ylim(y_lower, y_upper)

            for region_name, color in (
                ("clean_water", "tab:blue"),
                ("amd_water", "tab:orange"),
            ):
                outlier_dates = outlier_dates_by_region.get(region_name, [])
                if not outlier_dates:
                    continue
                axes[0].scatter(
                    outlier_dates,
                    np.repeat(y_lower + 0.02 * (y_upper - y_lower), len(outlier_dates)),
                    marker="x",
                    s=35,
                    alpha=0.8,
                    color=color,
                    label=f"{region_name} outlier removed",
                )

        for region_name, color, plot_df, lower_std, upper_std in plot_series:
            axes[1].plot(
                plot_df["date"],
                plot_df["mean"],
                linestyle="-",
                linewidth=1.6,
                color=color,
                label=f"{region_name} mean",
            )
            axes[1].fill_between(
                plot_df["date"],
                lower_std,
                upper_std,
                color=color,
                alpha=0.18,
                label=f"{region_name} mean +/- std",
            )

        handles, labels = axes[0].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if unique_labels:
            axes[0].legend(unique_labels.values(), unique_labels.keys(), loc="best")

        axes[0].set_title(f"Sampled Values After Outlier Removal: {variable}")
        axes[0].set_ylabel(variable)
        axes[0].grid(True, alpha=0.25)

        axes[1].set_title(f"Mean and Standard Deviation After Outlier Removal: {variable}")
        axes[1].set_xlabel("Date; x marks sampled outlier removed")
        axes[1].set_ylabel(variable)
        add_legend_if_needed(axes[1])
        axes[1].grid(True, alpha=0.25)

        fig.autofmt_xdate()
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"temporal_{variable}.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved temporal plot: {output_path}")


def main():
    project_dir = "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/"
    inputs_dir = os.path.join(project_dir, "inputs")
    tif_dir = os.path.join(inputs_dir, "sentinel2")
    cloud_dir = os.path.join(inputs_dir, "sentinel2_clouds")

    static_dir = os.path.join(project_dir, "static")
    clean_mask_path = os.path.join(static_dir, "yxsjoberg_clean_water_mask.tif")
    tsf_mask_path = os.path.join(static_dir, "yxsjoberg_tsf_water_mask.tif")

    output_dir = os.path.join(project_dir, "results", "eda")
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Sentinel-2 AMD monitoring EDA")
    print(f"[INFO] Raster directory: {tif_dir}")
    print(f"[INFO] Cloud-mask directory: {cloud_dir}")
    print(f"[INFO] Clean-water mask: {clean_mask_path}")
    print(f"[INFO] AMD-water mask: {tsf_mask_path}")
    print(f"[INFO] EDA output directory: {output_dir}")

    stats_df, histogram_values, temporal_values = build_eda_table(
        tif_dir=tif_dir,
        cloud_dir=cloud_dir,
        clean_mask_path=clean_mask_path,
        tsf_mask_path=tsf_mask_path,
    )

    summary_df = make_overall_summary(stats_df)

    print_results(stats_df, summary_df)
    save_overall_summary_json(summary_df, output_dir)
    plot_histograms(histogram_values, output_dir)
    plot_temporal_statistics(stats_df, temporal_values, output_dir)


if __name__ == "__main__":
    main()
