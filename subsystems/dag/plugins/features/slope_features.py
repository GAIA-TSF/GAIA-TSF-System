"""Causal dynamics and retrospective summaries for cumulative LOS rasters."""

from __future__ import annotations

from datetime import date

import numpy as np

from subsystems.dag.core.interfaces import FeatureExtractor
from subsystems.dag.utils.raster import temporal_mean, temporal_std
from subsystems.dag.utils.temporal import causal_temporal_gradient, linear_trend


class SlopeFeatureExtractor(FeatureExtractor):
    """Derive features from canonical cumulative LOS displacement ``C(t)``."""

    @property
    def name(self) -> str:
        return 'slope_feature_extractor'

    def compute(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
        enabled_features: dict[str, bool],
        summary_window: tuple[date, date] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return causal stacks and explicitly-windowed retrospective maps."""
        features: dict[str, np.ndarray] = {}
        if enabled_features.get('cumulative_los', False):
            features['cumulative_los'] = data.astype(np.float32, copy=True)
        if enabled_features.get('displacement_increment', False):
            features['displacement_increment'] = self.compute_displacement_increment(
                data
            )
        if enabled_features.get('velocity', False):
            features['velocity'] = self.compute_velocity(data, dates)
        if enabled_features.get('acceleration', False):
            features['acceleration'] = self.compute_acceleration(data, dates)
        if enabled_features.get('jerk', False):
            features['jerk'] = self.compute_jerk(data, dates)

        summary_options = (
            'minimum',
            'maximum',
            'mean',
            'standard_deviation',
            'variance',
            'range',
            'net_displacement',
            'trend',
            'temporal_variance',
        )
        if any(enabled_features.get(name, False) for name in summary_options):
            historical = self._summary_slice(data, dates, summary_window)
            features.update(self.compute_statistics(historical, enabled_features))
            if enabled_features.get('net_displacement', False):
                features['net_displacement'] = self._net_displacement(historical)
            if enabled_features.get('trend', False):
                features['temporal_summary_linear_trend'] = linear_trend(
                    historical, self._summary_dates(dates, summary_window)
                )
            if enabled_features.get('temporal_variance', False):
                features['temporal_summary_variance'] = self._variance_time(historical)
        return features

    @staticmethod
    def compute_displacement_increment(data: np.ndarray) -> np.ndarray:
        """Return causal displacement increments ``C(t) - C(t-1)``."""
        output = np.full(data.shape, np.nan, dtype=np.float32)
        output[1:] = data[1:] - data[:-1]
        return output

    @staticmethod
    def compute_velocity(data: np.ndarray, dates: tuple[date, ...]) -> np.ndarray:
        """Return causal velocity in displacement units per day."""
        return causal_temporal_gradient(data, dates, order=1)

    @staticmethod
    def compute_acceleration(data: np.ndarray, dates: tuple[date, ...]) -> np.ndarray:
        """Return causal acceleration in displacement units per day squared."""
        return causal_temporal_gradient(data, dates, order=2)

    @staticmethod
    def compute_jerk(data: np.ndarray, dates: tuple[date, ...]) -> np.ndarray:
        """Return optional causal jerk in displacement units per day cubed."""
        return causal_temporal_gradient(data, dates, order=3)

    def compute_statistics(
        self, data: np.ndarray, enabled_features: dict[str, bool]
    ) -> dict[str, np.ndarray]:
        """Return retrospective maps with names that cannot imply causality."""
        statistics: dict[str, np.ndarray] = {}
        mapping = {
            'minimum': ('temporal_summary_minimum', self._min_time),
            'maximum': ('temporal_summary_maximum', self._max_time),
            'mean': ('temporal_summary_mean', temporal_mean),
            'standard_deviation': ('temporal_summary_standard_deviation', temporal_std),
            'variance': ('temporal_summary_variance', self._variance_time),
        }
        for option, (name, function) in mapping.items():
            if enabled_features.get(option, False):
                statistics[name] = function(data)
        if enabled_features.get('range', False):
            statistics['temporal_summary_range'] = self._max_time(
                data
            ) - self._min_time(data)
        return statistics

    @staticmethod
    def _summary_slice(
        data: np.ndarray,
        dates: tuple[date, ...],
        summary_window: tuple[date, date] | None,
    ) -> np.ndarray:
        if summary_window is None:
            raise ValueError(
                'Retrospective summary products require an explicit summary_window.'
            )
        start, end = summary_window
        if start > end:
            raise ValueError('summary_window start must not be after its end.')
        indices = [index for index, value in enumerate(dates) if start <= value <= end]
        if not indices:
            raise ValueError('summary_window contains no acquisition dates.')
        return data[indices]

    @staticmethod
    def _summary_dates(
        dates: tuple[date, ...], summary_window: tuple[date, date] | None
    ) -> tuple[date, ...]:
        assert summary_window is not None
        start, end = summary_window
        return tuple(value for value in dates if start <= value <= end)

    @staticmethod
    def _net_displacement(data: np.ndarray) -> np.ndarray:
        output = np.full(data.shape[1:], np.nan, dtype=np.float32)
        for row in range(data.shape[1]):
            for column in range(data.shape[2]):
                values = data[:, row, column]
                finite = values[np.isfinite(values)]
                if finite.size:
                    output[row, column] = finite[-1] - finite[0]
        return output

    @staticmethod
    def _min_time(data: np.ndarray) -> np.ndarray:
        counts = np.sum(np.isfinite(data), axis=0)
        values = np.min(np.where(np.isfinite(data), data, np.inf), axis=0)
        return np.where(counts > 0, values, np.nan).astype(np.float32)

    @staticmethod
    def _max_time(data: np.ndarray) -> np.ndarray:
        counts = np.sum(np.isfinite(data), axis=0)
        values = np.max(np.where(np.isfinite(data), data, -np.inf), axis=0)
        return np.where(counts > 0, values, np.nan).astype(np.float32)

    @staticmethod
    def _variance_time(data: np.ndarray) -> np.ndarray:
        return np.square(temporal_std(data)).astype(np.float32)
