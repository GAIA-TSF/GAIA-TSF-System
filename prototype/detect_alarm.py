from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import rasterio

from config import Config


# ---------------------------------------------------------------------
# Constants (explicit class mapping)
# ---------------------------------------------------------------------
CLASS_MAP = {
    "AMD_WATER": 1,
    "WATER": 2,
}

NODATA = 255


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def compute_alarm_masks(
    amd: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Compute alarm masks.
    """

    amd_mask = amd > threshold

    amd_water_mask = pred == CLASS_MAP["AMD_WATER"]
    water_mask = pred == CLASS_MAP["WATER"]

    # alarms
    amd_water_alarm = amd_water_mask & amd_mask
    water_alarm = water_mask & amd_mask

    return {
        "amd_water_mask": amd_water_mask,
        "water_mask": water_mask,
        "amd_water_alarm": amd_water_alarm,
        "water_alarm": water_alarm,
    }


def compute_statistics(masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Compute pixel statistics."""
    return {
        "amd_water_px": int(np.sum(masks["amd_water_mask"])),
        "amd_water_alarm_px": int(np.sum(masks["amd_water_alarm"])),
        "water_px": int(np.sum(masks["water_mask"])),
        "water_alarm_px": int(np.sum(masks["water_alarm"])),
    }


# ---------------------------------------------------------------------
# Raster writing
# ---------------------------------------------------------------------
def write_alarm_raster(
    output_path: Path,
    mask_base: np.ndarray,
    mask_alarm: np.ndarray,
    profile: dict,
):
    """Write binary alarm raster."""
    out = np.full(mask_base.shape, NODATA, dtype="uint8")
    out[mask_base] = 0
    out[mask_alarm] = 1

    profile = profile.copy()
    profile.update(dtype="uint8", count=1, compress="lzw", nodata=NODATA)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)


def write_combined_raster(
    output_path: Path,
    masks: Dict[str, np.ndarray],
    profile: dict,
):
    """Write combined alarm raster."""
    combined = np.zeros(masks["amd_water_mask"].shape, dtype="uint8")

    combined[masks["water_alarm"]] = 1
    combined[masks["amd_water_alarm"]] = 2

    profile = profile.copy()
    profile.update(dtype="uint8", count=1, compress="lzw", nodata=None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(combined, 1)


# ---------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------
def process_scene(
    amd_path: Path,
    pred_path: Path,
    out_dir: Path,
    threshold: float,
) -> Dict:
    """Process single scene."""

    base = amd_path.stem.replace("_amd", "")

    with rasterio.open(amd_path) as amd_src, rasterio.open(pred_path) as pred_src:
        amd = amd_src.read(1)
        pred = pred_src.read(1)
        profile = amd_src.profile

    masks = compute_alarm_masks(amd, pred, threshold)
    stats = compute_statistics(masks)

    print(
        f"[INFO] {base} | AMD water: {stats['amd_water_alarm_px']}/{stats['amd_water_px']} "
        f"| water: {stats['water_alarm_px']}/{stats['water_px']}"
    )

    # outputs
    write_alarm_raster(
        out_dir / f"{base}_amdwater_alarm.tif",
        masks["amd_water_mask"],
        masks["amd_water_alarm"],
        profile,
    )

    write_alarm_raster(
        out_dir / f"{base}_water_alarm.tif",
        masks["water_mask"],
        masks["water_alarm"],
        profile,
    )

    write_combined_raster(
        out_dir / f"{base}_alarm_combined.tif",
        masks,
        profile,
    )

    return {"scene": base, **stats}


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def detect_amd_water_alarm(
    cfg: Config,
    threshold: float = 1.0,
):
    amd_dir = cfg.outputs_dir / "amd"
    pred_dir = cfg.outputs_dir / "predictions"
    out_dir = cfg.outputs_dir / "alarm"

    out_dir.mkdir(parents=True, exist_ok=True)

    amd_files = sorted(amd_dir.glob("*_amd.tif"))

    results: List[Dict] = []

    for amd_path in amd_files:
        base = amd_path.stem.replace("_amd", "")
        pred_path = pred_dir / f"{base}_pred.tif"

        if not pred_path.exists():
            print(f"[WARNING] Missing prediction: {base}")
            continue

        stats = process_scene(amd_path, pred_path, out_dir, threshold)
        results.append(stats)

    # save CSV
    df = pd.DataFrame(results)
    csv_path = out_dir / "alarm_statistics.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n[INFO] CSV saved: {csv_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main(config_path: Path = Path("config.yaml"), threshold: float = 2.0):
    cfg = Config(config_path)
    detect_amd_water_alarm(cfg, threshold)


if __name__ == "__main__":
    main()
    