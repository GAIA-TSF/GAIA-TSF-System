from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_feature_raster(path: str | Path) -> np.ndarray:
    """Load a feature raster from disk.

    Args:
        path: Path to a supported feature file.

    Returns:
        A NumPy array containing the feature values.
    """
    feature_path = Path(path)
    suffix = feature_path.suffix.lower()

    if suffix == ".npy":
        return np.load(feature_path)
    if suffix == ".npz":
        data = np.load(feature_path)
        return data[data.files[0]]
    if suffix in {".tif", ".tiff"}:
        raise NotImplementedError("GeoTIFF support requires rasterio.")
    raise ValueError(f"Unsupported feature format: {feature_path}")
