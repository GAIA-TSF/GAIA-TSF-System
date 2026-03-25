from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio

from config import Config


def list_scenes(folder: Path, sort_by_time: bool = True):
    scenes = sorted(folder.glob("*.tif"))

    if sort_by_time:
        scenes = sorted(scenes, key=lambda p: p.stem)

    return scenes

# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def estimate_cloud_coverage(scene_path: Path) -> float:
    """
    Estimate cloud coverage (%) using a simple threshold on band 2.

    NOTE: Placeholder logic – replace with proper cloud mask later.
    """
    try:
        with rasterio.open(scene_path) as src:
            band = src.read(2)

        cloud_mask = band < 1000  # TODO: improve thresholding
        cloud_fraction = np.sum(cloud_mask) / cloud_mask.size

        return cloud_fraction * 100

    except Exception as e:
        print(f"[WARNING] Failed to process {scene_path.name}: {e}")
        return np.nan


def filter_cloudless_scenes(
    scenes: List[Path], cloud_threshold: float
) -> Tuple[List[Path], List[float]]:
    """
    Filter scenes based on cloud coverage threshold.

    Returns:
        - filtered scene paths
        - cloud cover values
    """
    cloud_cover = []

    for scene in scenes:
        cc = estimate_cloud_coverage(scene)
        cloud_cover.append(cc)

    cloud_cover = np.array(cloud_cover)

    valid_idx = np.where(cloud_cover < cloud_threshold)[0]
    filtered_scenes = [scenes[i] for i in valid_idx]

    return filtered_scenes, cloud_cover.tolist()


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def print_summary(
    total: int, kept: int, threshold: float
) -> None:
    print("\n[INFO] Cloud filtering summary")
    print(f"  Threshold: {threshold}%")
    print(f"  Total scenes: {total}")
    print(f"  Kept: {kept}")
    print(f"  Discarded: {total - kept}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main(config_path: Path = Path("config.yaml"), threshold: float = 10.0):
    cfg = Config(config_path)

    # load scenes
    scenes = list_scenes(cfg.rasters_dir)

    if not scenes:
        raise RuntimeError(f"No scenes found in {cfg.rasters_dir}")

    # filter
    filtered_scenes, cloud_cover = filter_cloudless_scenes(
        scenes, threshold
    )

    # report
    print_summary(len(scenes), len(filtered_scenes), threshold)

    print("\n[INFO] Filtered scenes:")
    for s in filtered_scenes:
        print(f"  - {s.name}")

    return filtered_scenes


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main(threshold=5.0) 
