"""Bounded interpolation and robust date-aware smoothing for AMD EDA."""

from datetime import date

import numpy as np


def fill_short_gaps(values, dates, max_gap_days=30, require_same_year=True):
    """Linearly fill missing candidate dates bracketed by valid observations."""
    values = np.asarray(values, dtype=float)
    if max_gap_days < 1 or len(values) != len(dates):
        raise ValueError('Trend values/dates must match and max_gap_days must be positive.')
    filled = values.copy()
    interpolated = np.zeros(values.shape, dtype=bool)
    gap_days = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    for left, right in zip(valid, valid[1:]):
        missing = np.arange(left + 1, right)
        duration = (dates[right] - dates[left]).days
        if (not missing.size or duration > max_gap_days
                or (require_same_year and dates[left].year != dates[right].year)):
            continue
        elapsed = np.asarray([(dates[i] - dates[left]).days for i in missing])
        filled[missing] = values[left] + elapsed / duration * (values[right] - values[left])
        interpolated[missing] = True
        gap_days[missing] = duration
    return filled, interpolated, gap_days


def robust_lowess(values, dates, interpolated=None, window_days=45,
                  robust_iterations=2, interpolated_weight=.5,
                  process_each_year=True):
    """Local-linear tricube smoother evaluated without extrapolation."""
    values = np.asarray(values, dtype=float)
    if (window_days < 1 or robust_iterations < 0
            or not 0 <= interpolated_weight <= 1 or len(values) != len(dates)):
        raise ValueError('Invalid robust LOWESS parameters or mismatched dates.')
    interpolated = (np.zeros(values.shape, dtype=bool) if interpolated is None
                    else np.asarray(interpolated, dtype=bool))
    output = np.full(values.shape, np.nan)
    ordinal = np.asarray([value.toordinal() for value in dates], dtype=float)
    groups = sorted({d.year for d in dates}) if process_each_year else [None]
    for year in groups:
        group = np.asarray([year is None or d.year == year for d in dates])
        observed = group & np.isfinite(values)
        # Smoothing must not become an implicit second gap-filling method.
        targets = np.flatnonzero(group & np.isfinite(values))
        valid_indices = np.flatnonzero(observed)
        if valid_indices.size < 2:
            continue
        lo, hi = valid_indices[0], valid_indices[-1]
        targets = targets[(targets >= lo) & (targets <= hi)]
        robust = np.ones(valid_indices.size)
        base = np.where(interpolated[valid_indices], interpolated_weight, 1.0)
        for iteration in range(robust_iterations + 1):
            fitted = np.full(targets.size, np.nan)
            x = ordinal[valid_indices]
            y = values[valid_indices]
            for j, target in enumerate(targets):
                distance = np.abs(x - ordinal[target])
                local = distance <= window_days
                if local.sum() < 2:
                    continue
                weight = (1 - (distance[local] / window_days) ** 3) ** 3
                weight *= base[local] * robust[local]
                design = np.column_stack((np.ones(local.sum()), x[local] - ordinal[target]))
                if np.count_nonzero(weight) < 2:
                    continue
                beta = np.linalg.lstsq(
                    design * np.sqrt(weight[:, None]), y[local] * np.sqrt(weight),
                    rcond=None,
                )[0]
                fitted[j] = beta[0]
            if iteration == robust_iterations:
                output[targets] = fitted
                break
            if np.isfinite(fitted).sum() < 2:
                output[targets] = fitted
                break
            at_observations = np.interp(x, ordinal[targets][np.isfinite(fitted)],
                                        fitted[np.isfinite(fitted)])
            residual = np.abs(y - at_observations)
            scale = 6 * np.median(residual)
            robust = ((1 - np.minimum(residual / scale, 1) ** 2) ** 2
                      if scale > 0 else np.ones_like(residual))
    return output


def process_trend_stack(data, dates, domain, config):
    """Apply filling and smoothing to pixels inside a spatial domain."""
    filled = np.full(data.shape, np.nan, dtype=np.float32)
    smoothed = np.full(data.shape, np.nan, dtype=np.float32)
    interpolation = np.zeros(data.shape, dtype=np.float32)
    gap_days = np.full(data.shape, np.nan, dtype=np.float32)
    for row, col in zip(*np.nonzero(domain)):
        series, flags, gaps = fill_short_gaps(
            data[:, row, col], dates, int(config.get('max_gap_days', 30)),
            bool(config.get('require_same_year', True)),
        )
        filled[:, row, col] = series
        interpolation[:, row, col] = flags
        gap_days[:, row, col] = gaps
        smoothed[:, row, col] = robust_lowess(
            series, dates, flags, int(config.get('window_days', 45)),
            int(config.get('robust_iterations', 2)),
            float(config.get('interpolated_weight', .5)),
            bool(config.get('process_each_year', True)),
        )
    return filled, smoothed, interpolation, gap_days
