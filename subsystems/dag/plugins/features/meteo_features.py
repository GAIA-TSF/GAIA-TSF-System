"""Causal precipitation and temperature feature engineering."""

from __future__ import annotations

from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np

from subsystems.dag.core.interfaces import FeatureExtractor


class MeteoFeatureExtractor(FeatureExtractor):
    """Create DA_R_01/DA_R_02 causal features from aligned daily raster stacks.

    Input arrays have shape ``(time, rows, columns)`` and are expected in
    millimetres (precipitation) and degrees Celsius (temperature).  Rolling
    windows use acquisition dates rather than a fixed number of observations,
    so the results remain correct when a day is absent from the input series.
    """

    PRECIPITATION_FEATURES = (
        'precipitation',
        'precip_7d',
        'precip_14d',
        'precip_30d',
        'precip_60d',
        'days_since_heavy_rain',
        'max_precip_7d',
    )
    TEMPERATURE_FEATURES = (
        'temperature_mean',
        'temperature_min',
        'temperature_max',
        'temp_7d_mean',
        'temp_30d_mean',
        'temperature_anomaly',
    )
    COLD_REGION_FEATURES = (
        'freeze_thaw',
        'freezing_degree_days',
        'thawing_degree_days',
    )

    @property
    def name(self) -> str:
        """Return the plugin registry name."""
        return 'meteo_feature_extractor'

    def compute(
        self,
        data: dict[str, np.ndarray],
        dates: tuple[date, ...],
        enabled_features: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute enabled precipitation and temperature features.

        ``enabled_features`` may contain feature-name booleans directly.  The
        cold-region features additionally require ``cold_regions.enabled`` (or
        the backwards-compatible top-level ``cold_regions: true``).  Optional
        parameters are ``heavy_rain_threshold`` (default 20 mm),
        ``freezing_point`` (default 0 C), and ``temperature_baseline``.  The
        latter can be a scalar, a spatial array, or is otherwise calculated as
        the per-pixel mean over the supplied period.
        """
        if not isinstance(data, dict):
            raise TypeError('Meteorological data must be a mapping of raster stacks.')
        self._validate_dates(dates)

        outputs: dict[str, np.ndarray] = {}
        requested = self._requested_features(enabled_features)

        if requested.intersection(self.PRECIPITATION_FEATURES):
            precipitation = self._stack(data, 'precipitation', dates)
            if 'precipitation' in requested:
                outputs['precipitation'] = precipitation.copy()
            for days in (7, 14, 30, 60):
                name = f'precip_{days}d'
                if name in requested:
                    outputs[name] = self._rolling(precipitation, dates, days, 'sum')
            if 'max_precip_7d' in requested:
                outputs['max_precip_7d'] = self._rolling(
                    precipitation, dates, 7, 'max'
                )
            if 'days_since_heavy_rain' in requested:
                threshold = float(enabled_features.get('heavy_rain_threshold', 20.0))
                if threshold < 0:
                    raise ValueError('heavy_rain_threshold must be non-negative.')
                outputs['days_since_heavy_rain'] = self._days_since_event(
                    precipitation, dates, threshold
                )

        temperature_features = (
            *self.TEMPERATURE_FEATURES,
            *self.COLD_REGION_FEATURES,
        )
        if requested.intersection(temperature_features):
            mean_features = {
                'temperature_mean',
                'temp_7d_mean',
                'temp_30d_mean',
                'temperature_anomaly',
                'freezing_degree_days',
                'thawing_degree_days',
            }
            mean = (
                self._stack(data, 'temperature_mean', dates)
                if requested.intersection(mean_features)
                else None
            )
            minimum = (
                self._stack(data, 'temperature_min', dates)
                if requested.intersection({'temperature_min', 'freeze_thaw'})
                else None
            )
            maximum = (
                self._stack(data, 'temperature_max', dates)
                if requested.intersection({'temperature_max', 'freeze_thaw'})
                else None
            )
            for name, stack in (
                ('temperature_mean', mean),
                ('temperature_min', minimum),
                ('temperature_max', maximum),
            ):
                if name in requested and stack is not None:
                    outputs[name] = stack.copy()
            if 'temp_7d_mean' in requested:
                assert mean is not None
                outputs['temp_7d_mean'] = self._rolling(mean, dates, 7, 'mean')
            if 'temp_30d_mean' in requested:
                assert mean is not None
                outputs['temp_30d_mean'] = self._rolling(mean, dates, 30, 'mean')
            if 'temperature_anomaly' in requested:
                assert mean is not None
                baseline = enabled_features.get('temperature_baseline')
                if baseline is None:
                    baseline_array = self._nanmean(mean)
                else:
                    baseline_array = np.asarray(baseline, dtype=np.float32)
                    try:
                        baseline_array = np.broadcast_to(baseline_array, mean.shape[1:])
                    except ValueError as exc:
                        raise ValueError(
                            'temperature_baseline must be scalar or match the '
                            'raster shape.'
                        ) from exc
                outputs['temperature_anomaly'] = (
                    mean - baseline_array[np.newaxis, ...]
                ).astype(np.float32)

            freezing_point = float(enabled_features.get('freezing_point', 0.0))
            if 'freeze_thaw' in requested:
                assert minimum is not None and maximum is not None
                valid = np.isfinite(minimum) & np.isfinite(maximum)
                outputs['freeze_thaw'] = np.where(
                    valid,
                    (minimum <= freezing_point) & (maximum > freezing_point),
                    np.nan,
                ).astype(np.float32)
            if 'freezing_degree_days' in requested:
                assert mean is not None
                valid_temperature = np.isfinite(mean)
                outputs['freezing_degree_days'] = np.where(
                    valid_temperature,
                    np.maximum(freezing_point - mean, 0.0),
                    np.nan,
                ).astype(np.float32)
            if 'thawing_degree_days' in requested:
                assert mean is not None
                valid_temperature = np.isfinite(mean)
                outputs['thawing_degree_days'] = np.where(
                    valid_temperature,
                    np.maximum(mean - freezing_point, 0.0),
                    np.nan,
                ).astype(np.float32)

        return outputs

    def _requested_features(self, config: dict[str, Any]) -> set[str]:
        requested = {
            name
            for name in (*self.PRECIPITATION_FEATURES, *self.TEMPERATURE_FEATURES)
            if config.get(name, False) is True
        }
        requested.update(
            name for name in self.COLD_REGION_FEATURES if config.get(name) is True
        )
        cold_config = config.get('cold_regions', False)
        cold_enabled = (
            cold_config.get('enabled', False)
            if isinstance(cold_config, dict)
            else cold_config is True
        )
        if cold_enabled:
            requested.update(
                name
                for name in self.COLD_REGION_FEATURES
                if config.get(name, True) is not False
            )
        return requested

    def _stack(
        self, data: dict[str, np.ndarray], name: str, dates: tuple[date, ...]
    ) -> np.ndarray:
        if name not in data:
            raise KeyError(f'Missing meteorological input stack: {name}')
        stack = np.asarray(data[name], dtype=np.float32)
        if stack.ndim != 3:
            raise ValueError(f'{name} must have shape (time, rows, columns).')
        if stack.shape[0] != len(dates):
            raise ValueError(f'{name} time dimension does not match dates.')
        return stack

    def _validate_dates(self, dates: tuple[date, ...]) -> None:
        if not dates:
            raise ValueError('At least one meteorological date is required.')
        if any(current <= previous for previous, current in pairwise(dates)):
            raise ValueError('Meteorological dates must be strictly chronological.')

    def _rolling(
        self,
        stack: np.ndarray,
        dates: tuple[date, ...],
        days: int,
        operation: str,
    ) -> np.ndarray:
        output = np.full(stack.shape, np.nan, dtype=np.float32)
        start = 0
        for index, current_date in enumerate(dates):
            while (current_date - dates[start]).days >= days:
                start += 1
            values = stack[start : index + 1]
            valid = np.isfinite(values)
            counts = np.sum(valid, axis=0)
            if operation == 'sum':
                result = np.nansum(values, axis=0)
            elif operation == 'mean':
                result = np.divide(
                    np.nansum(values, axis=0),
                    counts,
                    out=np.full(stack.shape[1:], np.nan, dtype=np.float32),
                    where=counts > 0,
                )
            elif operation == 'max':
                result = np.max(np.where(valid, values, -np.inf), axis=0)
            else:  # pragma: no cover - private callers use known operations
                raise ValueError(f'Unsupported rolling operation: {operation}')
            output[index] = np.where(counts > 0, result, np.nan)
        return output

    def _days_since_event(
        self,
        precipitation: np.ndarray,
        dates: tuple[date, ...],
        threshold: float,
    ) -> np.ndarray:
        output = np.full(precipitation.shape, np.nan, dtype=np.float32)
        last_event = np.full(precipitation.shape[1:], -1, dtype=np.int32)
        ordinal_dates = np.asarray([value.toordinal() for value in dates])
        for index, ordinal in enumerate(ordinal_dates):
            valid = np.isfinite(precipitation[index])
            event = valid & (precipitation[index] >= threshold)
            last_event[event] = ordinal
            has_event = valid & (last_event >= 0)
            output[index, has_event] = ordinal - last_event[has_event]
        return output

    def _nanmean(self, stack: np.ndarray) -> np.ndarray:
        counts = np.sum(np.isfinite(stack), axis=0)
        return np.divide(
            np.nansum(stack, axis=0),
            counts,
            out=np.full(stack.shape[1:], np.nan, dtype=np.float32),
            where=counts > 0,
        ).astype(np.float32)
