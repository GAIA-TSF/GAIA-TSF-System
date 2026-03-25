from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from config import Config


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
S2_WAVELENGTHS = [443, 490, 560, 665, 705, 740,
                  783, 842, 865, 945, 1610, 2190]


# ---------------------------------------------------------------------
# Raster visualization
# ---------------------------------------------------------------------
def show_natural_color(raster_path: Path, r=4, g=3, b=2, figsize=(10, 10)):
    """Display Sentinel-2 raster in natural color."""
    with rasterio.open(raster_path) as src:
        rgb = src.read([r, g, b]).astype("float32")

    rgb_norm = (rgb - np.nanmin(rgb)) / (np.nanmax(rgb) - np.nanmin(rgb))

    fig, ax = plt.subplots(figsize=figsize)
    rasterio.plot.show(rgb_norm, ax=ax, title=f"Natural color: {raster_path.name}")
    plt.show()


def show_rgb_with_polygons(
    raster_path: Path,
    shapefile_path: Path,
    column: str,
    figsize=(10, 10),
):
    """Display RGB raster with polygon overlay."""
    polygons = gpd.read_file(shapefile_path)

    with rasterio.open(raster_path) as src:
        rgb = src.read([4, 3, 2]).astype("float32")
        rgb_norm = rgb / np.nanmax(rgb)

        fig, ax = plt.subplots(figsize=figsize)
        rasterio.plot.show(rgb_norm, transform=src.transform, ax=ax)

        if polygons.crs != src.crs:
            polygons = polygons.to_crs(src.crs)

        polygons.plot(
            ax=ax,
            column=column,
            cmap="tab10",
            facecolor="none",
            linewidth=1.5,
            legend=True,
        )

        ax.set_title(f"RGB + polygons: {raster_path.name}")
        plt.show()


# ---------------------------------------------------------------------
# Spectral plots
# ---------------------------------------------------------------------
def load_spectral_data(
    csv_path: Optional[Path],
    spectra_dict: Optional[Dict],
    class_column: str,
) -> Dict[str, np.ndarray]:
    """Load spectral data from CSV or dictionary."""
    if csv_path:
        df = pd.read_csv(csv_path)
        return {
            c: df[df[class_column] == c].iloc[:, 1:].values
            for c in df[class_column].unique()
        }

    if spectra_dict:
        return spectra_dict

    raise ValueError("Provide csv_path or spectra_dict")


def plot_spectral_curves(
    csv_path: Optional[Path] = None,
    spectra_dict: Optional[Dict] = None,
    class_column: str = "class",
    ylim=(0, 4000),
    figsize=(12, 8),
):
    """Plot spectral envelopes and mean curves."""
    spectra = load_spectral_data(csv_path, spectra_dict, class_column)

    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(spectra) / 2)),
        ncols=2,
        figsize=figsize,
    )
    axes = axes.flatten()

    for ax, (cls, data) in zip(axes, spectra.items()):
        q5, q95 = np.nanpercentile(data, [5, 95], axis=0)
        mean = np.nanmean(data, axis=0)

        # envelope
        polygon = Polygon(
            np.vstack([
                np.hstack((S2_WAVELENGTHS, S2_WAVELENGTHS[::-1])),
                np.hstack((q5, q95[::-1]))
            ]).T
        )
        ax.add_collection(PatchCollection([polygon], alpha=0.3))

        ax.plot(S2_WAVELENGTHS, mean, "o-")
        ax.set_title(cls)
        ax.set_ylim(ylim)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")

    plt.tight_layout()
    plt.show()


def plot_spectral_quantiles(
    class_name: str,
    csv_path: Optional[Path] = None,
    spectra_dict: Optional[Dict] = None,
    class_column: str = "class",
    figsize=(8, 5),
):
    """Plot quantile envelopes for a single class."""
    spectra = load_spectral_data(csv_path, spectra_dict, class_column)
    data = spectra[class_name]

    q5, q25, q50, q75, q95 = np.nanpercentile(data, [5, 25, 50, 75, 95], axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(S2_WAVELENGTHS, q50, "o-")
    ax.fill_between(S2_WAVELENGTHS, q5, q95, alpha=0.15)
    ax.fill_between(S2_WAVELENGTHS, q25, q75, alpha=0.3)

    ax.set_title(f"{class_name} spectral quantiles")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")

    plt.show()


# ---------------------------------------------------------------------
# Spectral Angle Mapper (SAM)
# ---------------------------------------------------------------------
def spectral_angle_map(
    scene_path: Path,
    class_name: str,
    csv_path: Path,
    class_column: str = "class",
    vmax: float = 0.8,
    plot: bool = True,
):
    """Compute SAM map."""
    df = pd.read_csv(csv_path)
    spectra = df[df[class_column] == class_name].iloc[:, 1:].values

    ref = np.nanmedian(spectra, axis=0)

    with rasterio.open(scene_path) as src:
        scene = src.read().astype("float32")
        rows, cols = src.height, src.width

    pixels = scene.reshape(scene.shape[0], -1).T

    dot = np.dot(pixels, ref)
    norm_pixels = np.linalg.norm(pixels, axis=1)
    norm_ref = np.linalg.norm(ref)

    cos = np.clip(dot / (norm_pixels * norm_ref + 1e-10), -1, 1)
    sam = np.arccos(cos).reshape(rows, cols)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.clip(sam, 0, vmax), vmin=0, vmax=vmax)
        ax.set_title(f"SAM: {class_name}")
        ax.axis("off")
        plt.colorbar(im, ax=ax)
        plt.show()

    return sam


# ---------------------------------------------------------------------
# Example entry point
# ---------------------------------------------------------------------
def main(config_path: Path = Path("config.yaml")):
    cfg = Config(config_path)

    csv_path = cfg.outputs_dir / "spectral_samples.csv"

    # plot_spectral_curves(csv_path=csv_path, ylim=(0, 12000))
    # plot_spectral_quantiles(class_name="crop", csv_path=csv_path)
    """
    spectral_angle_map(
        scene_path=cfg.rasters_dir / "20180701T102021.tif",
        class_name="crop",
        csv_path=csv_path,
    )
    """
    
if __name__ == "__main__":
    main()