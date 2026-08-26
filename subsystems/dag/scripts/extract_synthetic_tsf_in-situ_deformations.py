"""Create synthetic in-situ observations by sampling the TRUE_LOS rasters."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
DATE_PATTERN = re.compile(r"(20\d{6})")


def parse_date(filename: str) -> datetime:
    """Extract YYYYMMDD from a TRUE_LOS filename."""
    match = DATE_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot parse an acquisition date from {filename!r}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def load_observation_points(
    points_file: Path,
    target_crs: Any,
    label_column: str,
) -> dict[str, tuple[float, float]]:
    """Load every uniquely labelled point and transform it to the raster CRS."""
    if not points_file.exists():
        raise FileNotFoundError(f"Observation points file not found: {points_file}")
    frame = gpd.read_file(points_file)
    if label_column not in frame.columns:
        raise ValueError(f"{points_file} does not contain attribute {label_column!r}")
    if frame.crs is None:
        raise ValueError(f"{points_file} has no CRS")
    frame = frame.to_crs(target_crs)
    labels = frame[label_column].astype("string")
    if labels.isna().any() or labels.str.strip().eq("").any():
        raise ValueError(f"{points_file} contains an empty {label_column!r} value")
    duplicates = labels[labels.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate observation labels in {points_file}: {duplicates}")
    points: dict[str, tuple[float, float]] = {}
    for name, geometry in zip(labels, frame.geometry):
        if geometry is None or geometry.is_empty or geometry.geom_type != "Point":
            raise ValueError(f"Observation {name!r} must have a Point geometry")
        points[str(name)] = (float(geometry.x), float(geometry.y))
    if not points:
        raise ValueError(f"No observation points found in {points_file}")
    return points


def sample_neighbourhood(src: rasterio.io.DatasetReader, x: float, y: float, size: int) -> float:
    """Return the nodata-aware mean of an odd square pixel window around x/y."""
    if size < 1 or size % 2 == 0:
        raise ValueError("in_situ.sampling.window_size must be a positive odd integer")
    row, col = src.index(x, y)
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        raise ValueError(f"Observation ({x}, {y}) lies outside raster {src.name}")
    radius = size // 2
    values = src.read(
        1, window=Window(col - radius, row - radius, size, size), boundless=True, masked=True
    )
    return float(values.mean()) if values.count() else float("nan")


def extract(
    config_path: Path,
    project_dir: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    settings = config.get("in_situ")
    if not isinstance(settings, dict):
        raise ValueError("in_situ must be a mapping in the DAG configuration")
    if project_dir is None:
        root = config.get("project_dir")
        if not root:
            raise ValueError("project_dir is required when --project-dir is not supplied")
        project_dir = Path(root).expanduser().resolve()
    else:
        project_dir = project_dir.expanduser().resolve()
    inputs = settings.get("inputs", {})
    true_los_dir = project_dir / inputs.get("directory", "inputs/true_los")
    files = sorted(true_los_dir.glob(inputs.get("filename_pattern", "*.tif")))
    if not files:
        raise FileNotFoundError(f"No GeoTIFFs found in {true_los_dir}")

    static = settings.get("static", {})
    points_file = project_dir / static.get(
        "observation_points", "static/observation_points.gpkg"
    )
    output_csv = output_csv or project_dir / settings.get(
        "output_csv", "inputs/in_situ_deformation.csv"
    )
    sampling = settings.get("sampling", {})
    window_size = int(sampling.get("window_size", 3))
    noise_std = float(sampling.get("sensor_noise_std_mm", 1.0))
    platform = str(settings.get("platform", "OFFICE_REVIEW"))
    qc = int(settings.get("qc", 0))
    rng = np.random.default_rng(int(config.get("global", {}).get("random_seed", 42)))

    with rasterio.open(files[0]) as first:
        if first.crs is None:
            raise ValueError(f"Raster has no CRS: {files[0]}")
        raster_crs = first.crs
        points = load_observation_points(
            points_file,
            raster_crs,
            str(static.get("label_column", "label")),
        )
        transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        latlon = {
            name: tuple(reversed(transformer.transform(x, y)))
            for name, (x, y) in points.items()
        }

    rows: list[dict[str, object]] = []
    for tif in files:
        acquisition = parse_date(tif.name)
        timestamp = acquisition.strftime("%Y-%m-%dT10:00:00Z")
        with rasterio.open(tif) as src:
            if src.crs != raster_crs:
                raise ValueError(f"Raster CRS differs from the first acquisition: {tif}")
            for name, (x, y) in points.items():
                value_mm = sample_neighbourhood(src, x, y, window_size) * 1000.0
                if np.isfinite(value_mm) and noise_std:
                    value_mm += rng.normal(0.0, noise_std)
                lat, lon = latlon[name]
                rows.append(
                    {
                        "PLATFORM": platform,
                        "DATE": timestamp,
                        "LATITUDE": round(lat, 6),
                        "LONGITUDE": round(lon, 6),
                        "LOS_DEFORMATION": (
                            round(value_mm, 2) if np.isfinite(value_mm) else np.nan
                        ),
                        "QC": qc,
                    }
                )
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.config, args.project_dir, args.output)
    print(f"Written {len(result)} records")


if __name__ == "__main__":
    main()
