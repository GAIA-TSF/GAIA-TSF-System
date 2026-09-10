from __future__ import annotations

from datetime import date

import numpy as np

from subsystems.dag.core.interfaces import FeatureExtractor
from subsystems.dag.utils.raster import temporal_mean, temporal_std
from subsystems.dag.utils.temporal import linear_trend, nanmean_time, temporal_gradient


class SlopeFeatureExtractor(FeatureExtractor):
    """Compute deformation features from Sentinel-1 LOS time series."""

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return 'slope_feature_extractor'

    def compute(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
        enabled_features: dict[str, bool],
    ) -> dict[str, np.ndarray]:
        """Compute enabled slope-stability deformation features."""
        features: dict[str, np.ndarray] = {}

        if enabled_features.get('cumulative_displacement', False):
            features['cumulative_displacement'] = self._sum_time(data)
        if enabled_features.get('velocity', False):
            features['velocity'] = self.compute_velocity(data, dates)
        if enabled_features.get('acceleration', False):
            features['acceleration'] = self.compute_acceleration(data, dates)
        if enabled_features.get('jerk', False):
            features['jerk'] = self.compute_jerk(data, dates)

        features.update(self.compute_statistics(data, enabled_features))

        if enabled_features.get('trend', False):
            features['trend'] = linear_trend(data, dates)
        if enabled_features.get('temporal_variance', False):
            features['temporal_variance'] = self._variance_time(data)

        return features

    def compute_velocity(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
    ) -> np.ndarray:
        """Compute mean first temporal derivative in displacement units per day."""
        return nanmean_time(temporal_gradient(data, dates, order=1))

    def compute_acceleration(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
    ) -> np.ndarray:
        """Compute mean second temporal derivative in units per day squared."""
        return nanmean_time(temporal_gradient(data, dates, order=2))

    def compute_jerk(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
    ) -> np.ndarray:
        """Compute mean third temporal derivative in units per day cubed."""
        return nanmean_time(temporal_gradient(data, dates, order=3))

    def compute_statistics(
        self,
        data: np.ndarray,
        enabled_features: dict[str, bool],
    ) -> dict[str, np.ndarray]:
        """Compute enabled per-pixel temporal statistics."""
        statistics: dict[str, np.ndarray] = {}

        if enabled_features.get('minimum', False):
            statistics['minimum'] = self._min_time(data)
        if enabled_features.get('maximum', False):
            statistics['maximum'] = self._max_time(data)
        if enabled_features.get('mean', False):
            statistics['mean'] = temporal_mean(data)
        if enabled_features.get('standard_deviation', False):
            statistics['standard_deviation'] = temporal_std(data)
        if enabled_features.get('variance', False):
            statistics['variance'] = self._variance_time(data)
        if enabled_features.get('range', False):
            statistics['range'] = self._max_time(data) - self._min_time(data)

        return statistics

    def _sum_time(self, data: np.ndarray) -> np.ndarray:
        counts = np.sum(np.isfinite(data), axis=0)
        sums = np.nansum(data, axis=0)
        return np.where(counts > 0, sums, np.nan).astype(np.float32)

    def _min_time(self, data: np.ndarray) -> np.ndarray:
        counts = np.sum(np.isfinite(data), axis=0)
        values = np.where(np.isfinite(data), data, np.inf)
        minimum = np.min(values, axis=0)
        return np.where(counts > 0, minimum, np.nan).astype(np.float32)

    def _max_time(self, data: np.ndarray) -> np.ndarray:
        counts = np.sum(np.isfinite(data), axis=0)
        values = np.where(np.isfinite(data), data, -np.inf)
        maximum = np.max(values, axis=0)
        return np.where(counts > 0, maximum, np.nan).astype(np.float32)

    def _variance_time(self, data: np.ndarray) -> np.ndarray:
        std = temporal_std(data)
        return np.square(std).astype(np.float32)
