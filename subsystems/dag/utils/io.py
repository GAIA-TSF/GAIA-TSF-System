"""Writers for model-ready DAG feature rasters and JSON metadata."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from subsystems.dag.utils.raster import RasterProfile, write_raster


def write_feature_rasters(
    features: dict[str, np.ndarray],
    output_dir: Path,
    filenames: dict[str, str],
    profile: RasterProfile,
    raster_format: str,
    band_names: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Write feature arrays as independent rasters."""
    output_paths: dict[str, str] = {}
    for feature_name, values in features.items():
        filename = filenames.get(feature_name, f'{feature_name}.tif')
        output_path = output_dir / filename
        write_raster(output_path, values, profile, raster_format, band_names)
        output_paths[feature_name] = str(output_path)
    return output_paths


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
