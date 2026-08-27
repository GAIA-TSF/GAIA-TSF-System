from __future__ import annotations

from typing import Any

import numpy as np


def normalize_features(
    features: dict[str, np.ndarray],
    config: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Normalize feature arrays while preserving non-finite pixels.

    Normalization is disabled when the configuration is absent or when
    ``enabled`` is false. Constant-valued features become zero when enabled.
    """
    settings = config or {}
    if not isinstance(settings, dict):
        raise ValueError('preprocessing.normalization must be a mapping.')
    if not bool(settings.get('enabled', False)):
        return features
    method = str(settings.get('method', 'zscore')).lower()
    if method not in {'minmax', 'zscore'}:
        raise ValueError("Normalization method must be 'minmax' or 'zscore'.")
    per_feature = bool(settings.get('per_feature', True))

    shared_values = None
    if not per_feature:
        finite_parts = [values[np.isfinite(values)] for values in features.values()]
        populated = [values for values in finite_parts if values.size]
        if not populated:
            raise ValueError('Feature set contains no finite values to normalize.')
        shared_values = np.concatenate(populated)

    normalized: dict[str, np.ndarray] = {}
    for name, values in features.items():
        array = values.astype(np.float64, copy=False)
        finite = np.isfinite(array)
        reference = array[finite] if per_feature else shared_values
        if reference is None or not reference.size:
            normalized[name] = array.astype(np.float32)
            continue
        if method == 'minmax':
            offset = float(np.min(reference))
            scale = float(np.max(reference) - offset)
        else:
            offset = float(np.mean(reference))
            scale = float(np.std(reference))
        result = np.full(array.shape, np.nan, dtype=np.float64)
        result[finite] = (array[finite] - offset) / scale if scale > 0 else 0.0
        normalized[name] = result.astype(np.float32)
    return normalized
