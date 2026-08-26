from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from subsystems.dag.core.interfaces import Plugin


class TopographicFeatureExtractor(Plugin):
    """Derive static terrain features from a DEM."""

    @property
    def name(self) -> str:
        return 'topographic_feature_extractor'

    def compute(
        self,
        dem: np.ndarray,
        pixel_size_x: float,
        pixel_size_y: float,
        pi_window_size: int,
    ) -> dict[str, np.ndarray]:
        if dem.ndim != 2:
            raise ValueError('DEM must be a two-dimensional raster.')
        if pixel_size_x <= 0 or pixel_size_y <= 0:
            raise ValueError('DEM pixel sizes must be positive.')
        if pi_window_size < 3 or pi_window_size % 2 == 0:
            raise ValueError('PI window size must be an odd integer of at least 3.')

        elevation = dem.astype(np.float64, copy=False)
        valid = np.isfinite(elevation)
        if not np.any(valid):
            raise ValueError('DEM contains no finite elevation values.')

        local_sum = uniform_filter(
            np.where(valid, elevation, 0.0),
            size=pi_window_size,
            mode='nearest',
        )
        local_weight = uniform_filter(
            valid.astype(np.float64),
            size=pi_window_size,
            mode='nearest',
        )
        local_mean = np.divide(
            local_sum,
            local_weight,
            out=np.full_like(elevation, np.nan),
            where=local_weight > 0,
        )

        filled = np.where(valid, elevation, local_mean)
        dz_dy, dz_dx = np.gradient(filled, pixel_size_y, pixel_size_x)
        slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
        pi = elevation - local_mean

        return {
            'dem': np.where(valid, elevation, np.nan).astype(np.float32),
            'slope': np.where(valid, slope, np.nan).astype(np.float32),
            'pi': np.where(valid, pi, np.nan).astype(np.float32),
        }
