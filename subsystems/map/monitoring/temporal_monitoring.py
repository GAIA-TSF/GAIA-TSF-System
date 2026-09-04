"""Aggregate residual monitoring algorithms for the MAP dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemporalMonitoringResult:
    """Time-series monitoring signals derived from spatial residual products."""

    observed_mean: np.ndarray
    predicted_mean: np.ndarray
    uncertainty_mean: np.ndarray | None
    residual_mean: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    anomaly_magnitude: np.ndarray
    anomaly_threshold: float
    acceleration_cusum: np.ndarray
    deceleration_cusum: np.ndarray
    oscillation: np.ndarray
    persistent_acceleration: np.ndarray
    dynamics: np.ndarray
    regime_risk: np.ndarray
    medium_risk_threshold: float
    high_risk_threshold: float


class TemporalResidualMonitor:
    """Derive residual anomalies and physical acceleration warnings for a TSF."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Configure monitoring thresholds and smoothing from a mapping."""
        self.anomaly_threshold = self._positive(config, 'anomaly_magnitude_threshold')
        cusum = self._section(config, 'cusum')
        self.cusum_reference = self._non_negative(cusum, 'reference_value')
        self.cusum_decision = self._positive(cusum, 'decision_threshold')
        self.instability_direction = self._direction(cusum, 'instability_direction')
        self.cusum_signal = self._cusum_signal(cusum)
        self.smoothing_span = self._positive_integer(cusum, 'smoothing_span')
        self.persistence_window = self._positive_integer(cusum, 'persistence_window')
        self.persistence_threshold = self._unit_interval(
            cusum,
            'persistence_threshold',
        )
        regime = self._section(config, 'regime')
        self.regime_signal = self._regime_signal(regime)
        self.risk_smoothing_span = self._positive_integer(regime, 'smoothing_span')
        self.medium_risk_threshold = self._unit_interval(
            regime, 'medium_risk_threshold'
        )
        self.high_risk_threshold = self._unit_interval(regime, 'high_risk_threshold')
        if self.medium_risk_threshold >= self.high_risk_threshold:
            raise ValueError('monitoring.dashboard.regime thresholds must increase.')

    def analyze(
        self,
        observed_stack: np.ndarray,
        prediction_stack: np.ndarray,
        dates: tuple[str, ...],
        calibration_window: tuple[int, int],
        monitoring_window: tuple[int, int],
        uncertainty_stack: np.ndarray | None = None,
    ) -> TemporalMonitoringResult:
        """Analyze mean TSF residuals and calibrated acceleration behaviour.

        Args:
            observed_stack: Observations shaped ``(time, rows, columns)``.
            prediction_stack: Model predictions with the same shape.
            dates: ISO acquisition dates corresponding to stack time indices.
            calibration_window: Inclusive/exclusive calibration index bounds.
            monitoring_window: Inclusive/exclusive monitoring index bounds.
            uncertainty_stack: Optional prediction uncertainty stack.

        Returns:
            Aggregate residual and early-warning signals for all acquisitions.
        """
        self._validate_inputs(
            observed_stack,
            prediction_stack,
            dates,
            calibration_window,
            monitoring_window,
            uncertainty_stack,
        )
        observed_mean = self._spatial_mean(observed_stack)
        predicted_mean = self._spatial_mean(prediction_stack)
        uncertainty_mean = (
            None if uncertainty_stack is None else self._spatial_mean(uncertainty_stack)
        )
        residual_mean = observed_mean - predicted_mean
        anomaly_magnitude = np.abs(residual_mean)
        time_days = self._days_from_start(dates)
        cusum_values = (
            observed_mean if self.cusum_signal == 'observed_velocity' else residual_mean
        )
        acceleration = self._gradient(cusum_values, time_days)
        trend = self._ema(acceleration, self.smoothing_span)

        calibration_start, calibration_end = calibration_window
        baseline = trend[calibration_start:calibration_end]
        baseline = baseline[np.isfinite(baseline)]
        if baseline.size < 3:
            raise ValueError(
                'Calibration period contains too few valid residual samples.'
            )
        baseline_std = max(float(np.std(baseline)), np.finfo(np.float64).eps)
        directional_trend = trend * self.instability_direction
        directional_baseline = directional_trend[calibration_start:calibration_end]
        directional_baseline = directional_baseline[np.isfinite(directional_baseline)]
        directional_mean = float(np.mean(directional_baseline))
        zscore = (directional_trend - directional_mean) / baseline_std
        persistence = self._sign_persistence(zscore, self.persistence_window)

        acceleration_cusum, deceleration_cusum = self._cusum(
            zscore,
            monitoring_window[0],
        )
        monitoring_mask = self._window_mask(zscore.size, monitoring_window)
        oscillation = monitoring_mask & (persistence < self.persistence_threshold)
        persistent_acceleration = (
            monitoring_mask
            & (acceleration_cusum > self.cusum_decision)
            & (persistence >= self.persistence_threshold)
        )
        persistent_deceleration = (
            monitoring_mask
            & (deceleration_cusum > self.cusum_decision)
            & (persistence >= self.persistence_threshold)
        )
        dynamics = np.full(zscore.size, 'stable', dtype='<U12')
        dynamics[persistent_deceleration] = 'decelerating'
        dynamics[persistent_acceleration] = 'accelerating'
        # Regime evidence is deliberately model-relative.  Raw physical
        # acceleration contains the expected annual cycle, so standardising it
        # directly against a single calibration mean incorrectly flags the same
        # seasonal curvature when it recurs in later years.  Comparing the
        # observed acceleration with the baseline model's acceleration removes
        # that expected behaviour before testing for an unexpected shift.
        if self.regime_signal == 'unexpected_acceleration':
            expected_acceleration = self._gradient(predicted_mean, time_days)
            expected_trend = self._ema(expected_acceleration, self.smoothing_span)
            regime_signal = (trend - expected_trend) * self.instability_direction
        else:
            regime_signal = directional_trend
        regime_baseline = regime_signal[calibration_start:calibration_end]
        regime_baseline = regime_baseline[np.isfinite(regime_baseline)]
        if regime_baseline.size < 3:
            raise ValueError(
                'Calibration period contains too few valid baseline samples.'
            )
        regime_std = max(float(np.std(regime_baseline)), np.finfo(np.float64).eps)
        regime_zscore = (regime_signal - float(np.mean(regime_baseline))) / regime_std
        regime_persistence = self._sign_persistence(
            regime_zscore,
            self.persistence_window,
        )
        positive_shift = np.maximum(0.0, regime_zscore)
        instantaneous_risk = (1.0 - np.exp(-positive_shift)) * regime_persistence
        regime_risk = self._ema(instantaneous_risk, self.risk_smoothing_span)
        regime_risk = np.where(monitoring_mask, regime_risk, 0.0)
        return TemporalMonitoringResult(
            observed_mean=observed_mean,
            predicted_mean=predicted_mean,
            uncertainty_mean=uncertainty_mean,
            residual_mean=residual_mean,
            velocity=observed_mean,
            acceleration=acceleration,
            anomaly_magnitude=anomaly_magnitude,
            anomaly_threshold=self.anomaly_threshold,
            acceleration_cusum=acceleration_cusum,
            deceleration_cusum=deceleration_cusum,
            oscillation=oscillation,
            persistent_acceleration=persistent_acceleration,
            dynamics=dynamics,
            regime_risk=regime_risk,
            medium_risk_threshold=self.medium_risk_threshold,
            high_risk_threshold=self.high_risk_threshold,
        )

    def spatial_persistent_acceleration(
        self,
        residual_stack: np.ndarray,
        dates: tuple[str, ...],
        calibration_window: tuple[int, int],
        monitoring_window: tuple[int, int],
        persistence: int,
    ) -> np.ndarray:
        """Return per-pixel persistent directional CUSUM acceleration flags.

        This is a residual-based spatial diagnostic. The aggregate dashboard
        normally uses observed-velocity acceleration, because residual changes
        describe model error rather than physical deformation acceleration.

        Args:
            residual_stack: Observation-minus-prediction residual stack.
            dates: ISO acquisition dates.
            calibration_window: Inclusive/exclusive calibration index bounds.
            monitoring_window: Inclusive/exclusive monitoring index bounds.
            persistence: Consecutive CUSUM acceleration acquisitions required.

        Returns:
            Boolean stack shaped like ``residual_stack``. Only persistent
            acceleration during the monitoring window is true.
        """
        if persistence < 1:
            raise ValueError('Spatial CUSUM persistence must be at least one.')
        if residual_stack.ndim != 3 or residual_stack.shape[0] != len(dates):
            raise ValueError('Residual stack and acquisition dates are incompatible.')
        self._validate_window_bounds(
            residual_stack.shape[0],
            calibration_window,
            monitoring_window,
        )
        time_days = self._days_from_start(dates)
        filled = self._fill_temporal_gaps(residual_stack)
        rate = np.gradient(filled, time_days, axis=0, edge_order=1)
        directional_rate = self._ema_stack(
            rate * self.instability_direction,
            self.smoothing_span,
        )
        calibration = directional_rate[calibration_window[0] : calibration_window[1]]
        finite_calibration = np.isfinite(calibration)
        calibration_count = np.sum(finite_calibration, axis=0)
        baseline_mean = np.divide(
            np.nansum(calibration, axis=0),
            calibration_count,
            out=np.full(calibration.shape[1:], np.nan, dtype=np.float64),
            where=calibration_count > 0,
        )
        squared_deviation = np.where(
            finite_calibration,
            np.square(calibration - baseline_mean[np.newaxis, :, :]),
            0.0,
        )
        baseline_std = np.sqrt(
            np.divide(
                np.sum(squared_deviation, axis=0),
                calibration_count,
                out=np.full(calibration.shape[1:], np.nan, dtype=np.float64),
                where=calibration_count > 0,
            ),
        )
        valid_baseline = np.isfinite(baseline_mean) & (
            baseline_std > np.finfo(float).eps
        )
        zscore = np.divide(
            directional_rate - baseline_mean[np.newaxis, :, :],
            baseline_std[np.newaxis, :, :],
            out=np.full_like(directional_rate, np.nan),
            where=valid_baseline[np.newaxis, :, :],
        )
        output = np.zeros(residual_stack.shape, dtype=bool)
        cusum = np.zeros(residual_stack.shape[1:], dtype=np.float64)
        run = np.zeros(residual_stack.shape[1:], dtype=np.int16)
        for index in range(monitoring_window[0], monitoring_window[1]):
            values = zscore[index]
            valid = np.isfinite(values)
            cusum = np.where(
                valid,
                np.maximum(0.0, cusum + values - self.cusum_reference),
                cusum,
            )
            accelerating = valid & (cusum > self.cusum_decision)
            run = np.where(accelerating, run + 1, 0)
            output[index] = run >= persistence
        return output

    def _cusum(
        self, zscore: np.ndarray, monitoring_start: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one-sided positive and negative CUSUM signals."""
        positive = np.zeros(zscore.size, dtype=np.float64)
        negative = np.zeros(zscore.size, dtype=np.float64)
        for index in range(monitoring_start, zscore.size):
            value = zscore[index]
            if not np.isfinite(value):
                positive[index] = positive[index - 1] if index else 0.0
                negative[index] = negative[index - 1] if index else 0.0
                continue
            previous_positive = positive[index - 1] if index else 0.0
            previous_negative = negative[index - 1] if index else 0.0
            positive[index] = max(0.0, previous_positive + value - self.cusum_reference)
            negative[index] = max(0.0, previous_negative - value - self.cusum_reference)
        return positive, negative

    @staticmethod
    def _spatial_mean(values: np.ndarray) -> np.ndarray:
        """Return a finite-only mean for every acquisition."""
        finite = np.isfinite(values)
        count = np.sum(finite, axis=(1, 2))
        return np.divide(
            np.nansum(values, axis=(1, 2)),
            count,
            out=np.full(values.shape[0], np.nan, dtype=np.float64),
            where=count > 0,
        )

    @staticmethod
    def _gradient(values: np.ndarray, time_days: np.ndarray) -> np.ndarray:
        """Return an edge-safe temporal derivative after interpolating gaps."""
        finite = np.isfinite(values)
        if finite.sum() < 2:
            return np.full(values.shape, np.nan, dtype=np.float64)
        interpolated = np.interp(time_days, time_days[finite], values[finite])
        return np.gradient(interpolated, time_days, edge_order=1)

    @staticmethod
    def _ema(values: np.ndarray, span: int) -> np.ndarray:
        """Return an exponential moving average while carrying finite history."""
        output = np.empty(values.shape, dtype=np.float64)
        alpha = 2.0 / (span + 1.0)
        output[0] = values[0]
        for index in range(1, values.size):
            output[index] = (
                output[index - 1]
                if not np.isfinite(values[index])
                else alpha * values[index] + (1.0 - alpha) * output[index - 1]
            )
        return output

    @staticmethod
    def _ema_stack(values: np.ndarray, span: int) -> np.ndarray:
        """Return an exponential moving average for every raster pixel."""
        output = np.empty_like(values, dtype=np.float64)
        alpha = 2.0 / (span + 1.0)
        output[0] = values[0]
        for index in range(1, values.shape[0]):
            output[index] = np.where(
                np.isfinite(values[index]),
                alpha * values[index] + (1.0 - alpha) * output[index - 1],
                output[index - 1],
            )
        return output

    @staticmethod
    def _fill_temporal_gaps(values: np.ndarray) -> np.ndarray:
        """Linearly interpolate each pixel's gaps before temporal gradients."""
        time_count = values.shape[0]
        flattened = np.asarray(values, dtype=np.float64).reshape(time_count, -1)
        output = flattened.copy()
        positions = np.arange(time_count)
        for column in range(flattened.shape[1]):
            series = flattened[:, column]
            finite = np.isfinite(series)
            if finite.sum() >= 2:
                output[:, column] = np.interp(
                    positions,
                    positions[finite],
                    series[finite],
                )
        return output.reshape(values.shape)

    @staticmethod
    def _sign_persistence(values: np.ndarray, window: int) -> np.ndarray:
        """Measure local sign stability, where one indicates sustained motion."""
        signs = np.sign(np.nan_to_num(values, nan=0.0))
        signs[signs == 0.0] = 1.0
        changes = np.zeros(values.size, dtype=np.float64)
        changes[1:] = np.abs(np.diff(signs)) / 2.0
        change_rate = np.convolve(changes, np.ones(window) / window, mode='same')
        return np.clip(1.0 - change_rate, 0.0, 1.0)

    @staticmethod
    def _days_from_start(dates: tuple[str, ...]) -> np.ndarray:
        """Convert chronological ISO dates to floating day offsets."""
        parsed = [date.fromisoformat(value) for value in dates]
        return np.array([(value - parsed[0]).days for value in parsed], dtype=float)

    @staticmethod
    def _window_mask(length: int, window: tuple[int, int]) -> np.ndarray:
        """Return a Boolean mask for an exclusive index window."""
        mask = np.zeros(length, dtype=bool)
        mask[window[0] : window[1]] = True
        return mask

    @staticmethod
    def _validate_inputs(
        observed: np.ndarray,
        predicted: np.ndarray,
        dates: tuple[str, ...],
        calibration: tuple[int, int],
        monitoring: tuple[int, int],
        uncertainty: np.ndarray | None,
    ) -> None:
        """Validate common temporal monitoring input invariants."""
        if observed.ndim != 3 or observed.shape != predicted.shape:
            raise ValueError(
                'Observed and prediction stacks must be matching 3D arrays.'
            )
        if observed.shape[0] != len(dates):
            raise ValueError('Acquisition dates and temporal stacks are incompatible.')
        if uncertainty is not None and uncertainty.shape != observed.shape:
            raise ValueError('Uncertainty stack must match the observation stack.')
        TemporalResidualMonitor._validate_window_bounds(
            observed.shape[0],
            calibration,
            monitoring,
        )

    @staticmethod
    def _validate_window_bounds(
        time_count: int,
        calibration: tuple[int, int],
        monitoring: tuple[int, int],
    ) -> None:
        """Validate two exclusive temporal windows against a time dimension."""
        for name, window in (('calibration', calibration), ('monitoring', monitoring)):
            if not 0 <= window[0] < window[1] <= time_count:
                raise ValueError(f'{name} window is outside the acquisition range.')

    @staticmethod
    def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
        """Return a required monitoring configuration subsection."""
        value = config.get(name)
        if not isinstance(value, dict):
            raise ValueError(f'monitoring.dashboard.{name} must be a mapping.')
        return value

    @staticmethod
    def _positive(config: dict[str, Any], name: str) -> float:
        """Read a strictly positive scalar configuration value."""
        value = float(config[name])
        if value <= 0:
            raise ValueError(f'monitoring.dashboard.{name} must be positive.')
        return value

    @staticmethod
    def _non_negative(config: dict[str, Any], name: str) -> float:
        """Read a non-negative scalar configuration value."""
        value = float(config[name])
        if value < 0:
            raise ValueError(f'monitoring.dashboard.{name} must be non-negative.')
        return value

    @staticmethod
    def _positive_integer(config: dict[str, Any], name: str) -> int:
        """Read a strictly positive integer configuration value."""
        value = int(config[name])
        if value < 1:
            raise ValueError(f'monitoring.dashboard.{name} must be at least one.')
        return value

    @staticmethod
    def _unit_interval(config: dict[str, Any], name: str) -> float:
        """Read a scalar in the closed unit interval."""
        value = float(config[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'monitoring.dashboard.{name} must be in [0, 1].')
        return value

    @staticmethod
    def _direction(config: dict[str, Any], name: str) -> float:
        """Map a configured physical instability direction to a sign multiplier."""
        value = config[name]
        if value == 'positive':
            return 1.0
        if value == 'negative':
            return -1.0
        raise ValueError(
            'monitoring.dashboard.cusum.instability_direction must be '
            "'positive' or 'negative'.",
        )

    @staticmethod
    def _cusum_signal(config: dict[str, Any]) -> str:
        """Return the configured physical or residual CUSUM input series."""
        value = str(config.get('signal', 'observed_velocity'))
        if value not in {'observed_velocity', 'residual'}:
            raise ValueError(
                'monitoring.dashboard.cusum.signal must be '
                '"observed_velocity" or "residual".',
            )
        return value

    @staticmethod
    def _regime_signal(config: dict[str, Any]) -> str:
        """Return the configured model-relative or raw regime input series."""
        value = str(config.get('signal', 'unexpected_acceleration'))
        if value not in {'unexpected_acceleration', 'observed_acceleration'}:
            raise ValueError(
                'monitoring.dashboard.regime.signal must be '
                '"unexpected_acceleration" or "observed_acceleration".',
            )
        return value
