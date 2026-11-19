"""Minimal grid/date container used by engine B (replaces MAP's ``Dataset``).

The prototype passed MAP's ``Dataset`` around but only used its dates, TSF mask
and raster grid. ALE builds this from the hand-off contract instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridDataset:
    """Acquisition dates, TSF mask and raster georeferencing."""

    dates: tuple[str, ...]
    mask: np.ndarray
    crs: str = ''
    transform: tuple[float, ...] = ()
    nodata: float = float('nan')
    target_feature: str = ''

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])
