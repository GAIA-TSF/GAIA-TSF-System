"""Lag, difference, rolling, and smoothed temporal raster features."""

from __future__ import annotations

from datetime import date
import logging

import numpy as np

from subsystems.dag.core.interfaces import FeatureExtractor


LOGGER = logging.getLogger(__name__)


class TemporalFeatureExtractor(FeatureExtractor):
    """Compute generic temporal features from raster feature stacks."""

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return 'temporal_feature_extractor'

    def compute(
        self,
        data: dict[str, np.ndarray],
        dates: tuple[date, ...],
        enabled_features: dict[str, object],
    ) -> dict[str, np.ndarray]:
        """Compute enabled temporal features for each input feature stack.

        Args:
            data: Mapping of base feature name to temporal stack with shape
                ``(time, rows, cols)``.
            dates: Chronological acquisition dates.
            enabled_features: Temporal feature configuration.

        Returns:
            Mapping of output feature names to temporal raster stacks with shape
            ``(time, rows, cols)``.
        """
        outputs: dict[str, np.ndarray] = {}
        for feature_name, stack in data.items():
            self._validate_stack(feature_name, stack, dates)
            outputs.update(self._compute_lags(feature_name, stack, enabled_features))
            outputs.update(
                self._compute_differences(feature_name, stack, enabled_features),
            )
            outputs.update(
                self._compute_rolling_mean(feature_name, stack, enabled_features),
            )
            outputs.update(
                self._compute_rolling_std(feature_name, stack, enabled_features),
            )
            # no savgol smoothing
            # outputs.update(
            #     self._compute_smoothing(feature_name, stack, enabled_features),
            # )
        return outputs

    def _compute_lags(
        self,
        feature_name: str,
        stack: np.ndarray,
        config: dict[str, object],
    ) -> dict[str, np.ndarray]:
        lag_config = self._section(config, 'lag')
        if not lag_config.get('enabled', False):
            return {}

        return {
            f'{feature_name}_lag{order}': self._lag(stack, order)
            for order in self._orders(lag_config)
        }

    def _compute_differences(
        self,
        feature_name: str,
        stack: np.ndarray,
        config: dict[str, object],
    ) -> dict[str, np.ndarray]:
        difference_config = self._section(config, 'difference')
        if not difference_config.get('enabled', False):
            return {}

        return {
            f'{feature_name}_diff{order}': self._difference(stack, order)
            for order in self._orders(difference_config)
        }

    def _compute_rolling_mean(
        self,
        feature_name: str,
        stack: np.ndarray,
        config: dict[str, object],
    ) -> dict[str, np.ndarray]:
        rolling_config = self._section(config, 'rolling_mean')
        if not rolling_config.get('enabled', False):
            return {}

        window = self._window(rolling_config)
        return {
            f'{feature_name}_roll_mean': self._rolling_mean(stack, window),
        }

    def _compute_rolling_std(
        self,
        feature_name: str,
        stack: np.ndarray,
        config: dict[str, object],
    ) -> dict[str, np.ndarray]:
        rolling_config = self._section(config, 'rolling_std')
        if not rolling_config.get('enabled', False):
            return {}

        window = self._window(rolling_config)
        return {
            f'{feature_name}_roll_std': self._rolling_std(stack, window),
        }

    def _compute_smoothing(
        self,
        feature_name: str,
        stack: np.ndarray,
        config: dict[str, object],
    ) -> dict[str, np.ndarray]:
        smoothing_config = self._section(config, 'smoothing')
        if not smoothing_config.get('enabled', False):
            return {}

        method = str(smoothing_config.get('method', 'savgol'))
        if method != 'savgol':
            raise ValueError(f'Unsupported smoothing method: {method}')

        window = self._window(smoothing_config)
        polyorder = int(smoothing_config.get('polyorder', 2))
        return {
            f'{feature_name}_smooth': self._savgol_series(
                stack,
                window=window,
                polyorder=polyorder,
            ),
        }

    def _lag(self, stack: np.ndarray, order: int) -> np.ndarray:
        if order >= stack.shape[0]:
            raise ValueError(
                f'Lag order {order} requires more than {order} time steps.',
            )
        output = np.full(stack.shape, np.nan, dtype=np.float32)
        output[order:] = stack[:-order]
        return output

    def _difference(self, stack: np.ndarray, order: int) -> np.ndarray:
        if order >= stack.shape[0]:
            raise ValueError(
                f'Difference order {order} requires more than {order} time steps.',
            )
        output = np.full(stack.shape, np.nan, dtype=np.float32)
        output[order:] = stack[order:] - stack[:-order]
        return output

    def _rolling_mean(self, stack: np.ndarray, window: int) -> np.ndarray:
        self._validate_window_length(stack, window)
        output = np.full(stack.shape, np.nan, dtype=np.float32)
        for index in range(window - 1, stack.shape[0]):
            values = stack[index - window + 1 : index + 1]
            counts = np.sum(np.isfinite(values), axis=0)
            sums = np.nansum(values, axis=0)
            output[index] = np.divide(
                sums,
                counts,
                out=np.full(stack.shape[1:], np.nan, dtype=np.float32),
                where=counts > 0,
            )
        return output

    def _rolling_std(self, stack: np.ndarray, window: int) -> np.ndarray:
        self._validate_window_length(stack, window)
        output = np.full(stack.shape, np.nan, dtype=np.float32)
        for index in range(window - 1, stack.shape[0]):
            values = stack[index - window + 1 : index + 1]
            counts = np.sum(np.isfinite(values), axis=0)
            means = np.divide(
                np.nansum(values, axis=0),
                counts,
                out=np.full(stack.shape[1:], np.nan, dtype=np.float32),
                where=counts > 0,
            )
            squared_deviation = np.where(
                np.isfinite(values),
                np.square(values - means[np.newaxis, :, :]),
                0.0,
            )
            variance = np.divide(
                np.sum(squared_deviation, axis=0),
                counts,
                out=np.full(stack.shape[1:], np.nan, dtype=np.float32),
                where=counts > 0,
            )
            output[index] = np.sqrt(variance).astype(np.float32)
        return output

    def _savgol_series(
        self,
        stack: np.ndarray,
        window: int,
        polyorder: int,
    ) -> np.ndarray:
        if window > stack.shape[0]:
            self._validate_window_length(stack, window)
        if window % 2 == 0:
            raise ValueError('Savitzky-Golay window must be odd.')
        if polyorder >= window:
            raise ValueError('Savitzky-Golay polyorder must be smaller than window.')

        try:
            from scipy.signal import savgol_filter
        except ModuleNotFoundError:
            LOGGER.warning(
                'SciPy is unavailable; using NumPy polynomial Savitzky-Golay '
                'fallback for smoothed time series.',
            )
            return self._savgol_series_numpy(stack, window, polyorder)
        # TODO: remove

        # debug
        print(np.isnan(stack).sum())
        print(np.isinf(stack).sum())
        print(stack.shape)

        smoothed = savgol_filter(
            stack,
            window_length=window,
            polyorder=polyorder,
            axis=0,
            mode='interp',
        )
        return smoothed.astype(np.float32)

    def _savgol_series_numpy(
        self,
        stack: np.ndarray,
        window: int,
        polyorder: int,
    ) -> np.ndarray:
        time_steps = stack.shape[0]
        x = np.arange(time_steps, dtype=np.float64)
        stack = stack.astype(np.float64)
        valid = np.isfinite(stack)
        flattened = stack.reshape(time_steps, -1)
        flattened_valid = valid.reshape(time_steps, -1)
        output = np.full(flattened.shape, np.nan, dtype=np.float32)
        half_window = window // 2

        for index in range(flattened.shape[1]):
            for time_index in range(time_steps):
                start = min(max(time_index - half_window, 0), time_steps - window)
                end = start + window
                window_valid = flattened_valid[start:end, index]
                if np.count_nonzero(window_valid) <= polyorder:
                    continue
                coefficients = np.polyfit(
                    x[start:end][window_valid],
                    flattened[start:end, index][window_valid],
                    deg=polyorder,
                )
                output[time_index, index] = np.polyval(coefficients, x[time_index])

        return output.reshape(stack.shape).astype(np.float32)

    def _validate_window_length(self, stack: np.ndarray, window: int) -> None:
        if window > stack.shape[0]:
            raise ValueError(
                f'Temporal feature window {window} exceeds time steps '
                f'{stack.shape[0]}.',
            )

    def _section(
        self,
        config: dict[str, object],
        name: str,
    ) -> dict[str, object]:
        section = config.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f'Temporal feature config for {name} must be a mapping.')
        return section

    def _orders(self, config: dict[str, object]) -> list[int]:
        orders = config.get('orders', [])
        if not isinstance(orders, list) or not orders:
            raise ValueError('Temporal feature orders must be a non-empty list.')
        return [int(order) for order in orders]

    def _window(self, config: dict[str, object]) -> int:
        window = int(config.get('window', 0))
        if window <= 0:
            raise ValueError('Temporal feature window must be positive.')
        return window

    def _validate_stack(
        self,
        feature_name: str,
        stack: np.ndarray,
        dates: tuple[date, ...],
    ) -> None:
        if stack.ndim != 3:
            raise ValueError(
                f'Temporal feature stack {feature_name} must have shape '
                '(time, rows, cols).',
            )
        if stack.shape[0] != len(dates):
            raise ValueError(
                f'Temporal feature stack {feature_name} does not match dates.',
            )
