from __future__ import annotations

from typing import Any

import numpy as np


def transform_outliers(
    features: dict[str, np.ndarray],
    config: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Apply an opt-in log transform or quantile clipping to feature values."""
    settings = config or {}
    if not isinstance(settings, dict):
        raise ValueError('preprocessing.outliers must be a mapping.')
    if not bool(settings.get('enabled', False)):
        return features
    method = str(settings.get('method', 'log')).lower()
    if method not in {'log', 'clip'}:
        raise ValueError("Outlier method must be 'log' or 'clip'.")
    configured_names = settings.get('features', [])
    if not isinstance(configured_names, list):
        raise ValueError('preprocessing.outliers.features must be a list.')
    selected = {str(name) for name in configured_names}
    unknown = selected.difference(features)
    if unknown:
        raise ValueError(f'Configured outlier features do not exist: {sorted(unknown)}')
    selected = selected or set(features)

    clip_range = settings.get('clip_range', [0.01, 0.99])
    if method == 'clip':
        if not isinstance(clip_range, (list, tuple)) or len(clip_range) != 2:
            raise ValueError('preprocessing.outliers.clip_range must contain two values.')
        lower_quantile, upper_quantile = map(float, clip_range)
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError('Outlier clip quantiles must satisfy 0 <= low < high <= 1.')

    signed = bool(settings.get('signed_log', True))
    transformed: dict[str, np.ndarray] = {}
    for name, values in features.items():
        if name not in selected:
            transformed[name] = values
            continue
        array = values.astype(np.float64, copy=False)
        finite = np.isfinite(array)
        result = np.full(array.shape, np.nan, dtype=np.float64)
        if not np.any(finite):
            transformed[name] = result.astype(np.float32)
            continue
        if method == 'log':
            if not signed and np.any(array[finite] < 0):
                raise ValueError(
                    f"Feature {name!r} contains negative values; enable signed_log."
                )
            result[finite] = (
                np.sign(array[finite]) * np.log1p(np.abs(array[finite]))
                if signed
                else np.log1p(array[finite])
            )
        else:
            lower, upper = np.quantile(
                array[finite], [lower_quantile, upper_quantile]
            )
            result[finite] = np.clip(array[finite], lower, upper)
        transformed[name] = result.astype(np.float32)
    return transformed
