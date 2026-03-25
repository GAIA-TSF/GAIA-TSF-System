 
from pathlib import Path
from typing import List

import numpy as np
import rasterio

from config import Config


# ---------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------
def compute_amd_index(
    input_raster: Path,
    output_raster: Path,
    band_red: int = 4,
    band_blue: int = 2,
) -> None:
    """
    Compute AMD index (B4 / B2) from a multiband raster.
    """

    with rasterio.open(input_raster) as src:
        red = src.read(band_red).astype("float32")
        blue = src.read(band_blue).astype("float32")

        # safer division
        amd = np.divide(
            red,
            blue,
            out=np.zeros_like(red, dtype="float32"),
            where=blue != 0,
        )

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    output_raster.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(amd, 1)

    print(f"[INFO] AMD saved: {output_raster.name}")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def list_scenes(folder: Path) -> List[Path]:
    """List Sentinel-2 scenes."""
    if not folder.exists():
        raise FileNotFoundError(folder)

    return sorted(folder.glob("*.tif"))


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------
def process_amd_batch(
    scenes: List[Path],
    output_dir: Path,
) -> None:
    """Compute AMD index for multiple scenes."""
    for scene_path in scenes:
        base = scene_path.stem
        out_path = output_dir / f"{base}_amd.tif"

        print(f"[INFO] Processing {scene_path.name}")
        compute_amd_index(scene_path, out_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(config_path: Path = Path("config.yaml")):
    cfg = Config(config_path)

    scenes = list_scenes(cfg.rasters_dir)

    if not scenes:
        raise RuntimeError("No scenes found")

    amd_dir = cfg.outputs_dir / "amd"

    process_amd_batch(scenes, amd_dir)

    print(f"\n[INFO] Processed {len(scenes)} scenes")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main() 