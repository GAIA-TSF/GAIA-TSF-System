from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

from config import Config
# from data.inventory import list_shapefiles


def list_shapefiles(folder: Path, recursive: bool = False) -> List[Path]:
    """
    List shapefiles in directory.

    Parameters
    ----------
    recursive : bool
        Search subdirectories if True
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    pattern = "**/*.shp" if recursive else "*.shp"
    shp_files = sorted(folder.glob(pattern))

    if not shp_files:
        print(f"[WARNING] No shapefiles found in {folder}")

    return shp_files

# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def match_scene_to_shapefile(shp_path: Path, rasters_dir: Path) -> Path:
    """
    Match shapefile to corresponding Sentinel-2 scene.
    Assumes same basename.
    """
    scene_path = rasters_dir / f"{shp_path.stem}.tif"

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found for {shp_path.name}")

    return scene_path


def extract_spectra_from_polygons(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    class_field: str,
) -> Dict[str, np.ndarray]:
    """
    Extract spectra per class from one shapefile + raster.
    """
    spectra_dict = {}

    with rasterio.open(raster_path) as src:

        # reproject polygons if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        for class_name, group in gdf.groupby(class_field):

            shapes = group.geometry.values

            try:
                out_image, _ = mask(src, shapes, crop=True)
            except Exception as e:
                print(f"[WARNING] Mask failed for {class_name}: {e}")
                continue

            # reshape to (pixels x bands)
            pixels = out_image.reshape(out_image.shape[0], -1).T

            # filter invalid pixels
            pixels = pixels[np.all(pixels > 0, axis=1)]

            if pixels.size == 0:
                continue

            spectra_dict[class_name] = pixels

    return spectra_dict


def build_spectral_library(
    shp_paths: List[Path],
    cfg: Config,
) -> pd.DataFrame:
    """
    Build spectral library from shapefiles and matching scenes.
    """
    all_spectra = []

    for shp_path in shp_paths:
        print(f"[INFO] Processing {shp_path.name}")

        gdf = gpd.read_file(shp_path)

        scene_path = match_scene_to_shapefile(shp_path, cfg.rasters_dir)

        spectra_dict = extract_spectra_from_polygons(
            gdf,
            scene_path,
            cfg.class_field,
        )

        for class_name, pixels in spectra_dict.items():
            df = pd.DataFrame(pixels)
            df.insert(0, "class", class_name)
            all_spectra.append(df)

    if not all_spectra:
        raise RuntimeError("No spectra extracted.")

    return pd.concat(all_spectra, ignore_index=True)


def save_spectral_library(df: pd.DataFrame, output_path: Path) -> None:
    """Save spectral library to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.columns = ["class"] + [f"B{i+1}" for i in range(df.shape[1] - 1)]

    df.to_csv(output_path, index=False)

    print(f"[INFO] Saved spectral library: {output_path}")
    print(f"[INFO] Total samples: {len(df)}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main(config_path: Path = Path("config.yaml")):
    cfg = Config(config_path)

    shp_paths = list_shapefiles(cfg.shapefiles_dir)

    if not shp_paths:
        raise RuntimeError(f"No shapefiles found in {cfg.shapefiles_dir}")

    df = build_spectral_library(shp_paths, cfg)

    output_csv = cfg.outputs_dir / "spectral_samples.csv"

    save_spectral_library(df, output_csv)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main() 