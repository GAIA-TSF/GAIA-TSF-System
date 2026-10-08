"""Causal AMD history indexed by valid observations at each pixel."""

import numpy as np


def compute_amd_temporal(data, dates, config):
    window = config.get('rolling_window', 5)
    minimum = config.get('min_periods', window)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (window, minimum)) or minimum > window:
        raise ValueError('Require 1 <= min_periods <= rolling_window.')
    lags = config.get('lag_orders', [1, 2, 3])
    differences = config.get('difference_orders', [1, 2, 3])
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1
           for k in [*lags, *differences]):
        raise ValueError('Temporal orders must be positive integers.')
    calendar = config.get('calendar_features', ['annual_sin', 'annual_cos'])
    if set(calendar) - {'annual_sin', 'annual_cos'}:
        raise ValueError('Unsupported AMD calendar feature.')
    names = [*(f'lag{k}' for k in lags), *(f'amd_diff{k}' for k in differences),
             'roll_mean', 'roll_std', *calendar]
    outputs = {name: np.full(data.shape, np.nan, dtype=np.float32) for name in names}
    # Keep only the history needed, updating it after emitting causal lags.
    depth = max([window, *lags, *differences])
    history = np.full((depth, *data.shape[1:]), np.nan, dtype=np.float32)
    for t, day in enumerate(dates):
        valid = np.isfinite(data[t])
        for k in lags:
            outputs[f'lag{k}'][t, valid] = history[k - 1, valid]
        for k in differences:
            outputs[f'amd_diff{k}'][t, valid] = data[t, valid] - history[k - 1, valid]
        history[1:, valid] = history[:-1, valid].copy()
        history[0, valid] = data[t, valid]
        values = history[:window, valid]
        counts = np.isfinite(values).sum(axis=0)
        means = np.divide(np.nansum(values, axis=0), counts,
                          out=np.full(counts.shape, np.nan), where=counts >= minimum)
        variance = np.divide(np.nansum((values - means) ** 2, axis=0), counts,
                             out=np.full(counts.shape, np.nan), where=counts >= minimum)
        outputs['roll_mean'][t, valid] = means
        outputs['roll_std'][t, valid] = np.sqrt(variance)
        phase = 2 * np.pi * (day.timetuple().tm_yday - 1) / 365.2425
        for name in calendar:
            outputs[name][t, valid] = np.sin(phase) if name == 'annual_sin' else np.cos(phase)
    return outputs
