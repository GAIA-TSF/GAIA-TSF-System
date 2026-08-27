"""Opt-in imputation and consistent incomplete-sample removal."""

from __future__ import annotations

from typing import Any

import numpy as np


def handle_missing_values(
    features: dict[str, np.ndarray],
    config: dict[str, Any] | None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Implement DA_R_09 according to opt-in missing-value configuration.

    ``mean`` and ``median`` impute independently for each feature. ``drop``
    preserves raster dimensions but marks every incomplete sample position as
    missing across all features, allowing downstream dataset construction to
    remove those entries consistently.

    Args:
        features: Equally shaped named feature arrays.
        config: ``preprocessing.missing_values`` mapping containing
            ``enabled``, ``strategy``, and ``max_nan_ratio``.
        valid_mask: Optional spatial or spatiotemporal analysis-domain mask.
            Structural nodata outside this mask is neither counted nor imputed.

    Returns:
        The original mapping when disabled; otherwise, new ``float32`` arrays
        produced by the selected strategy.

    Raises:
        ValueError: If shapes or configuration are invalid, the permitted
            missing ratio is exceeded, or an all-missing feature cannot be
            imputed.
    """
    settings = config or {}
    if not isinstance(settings, dict):
        raise ValueError('preprocessing.missing_values must be a mapping.')
    if not bool(settings.get('enabled', False)):
        return features
    strategy = str(settings.get('strategy', 'median')).lower()
    if strategy not in {'mean', 'median', 'drop'}:
        raise ValueError("Missing-value strategy must be 'mean', 'median', or 'drop'.")
    max_missing_ratio = float(settings.get('max_nan_ratio', 1.0))
    if not 0.0 <= max_missing_ratio <= 1.0:
        raise ValueError('preprocessing.missing_values.max_nan_ratio must be in [0, 1].')
    if not features:
        return features

    shapes = {values.shape for values in features.values()}
    if len(shapes) != 1:
        raise ValueError('All features must have the same shape for missing-value handling.')
    arrays = {
        name: values.astype(np.float64, copy=False)
        for name, values in features.items()
    }
    shape = next(iter(shapes))
    if valid_mask is None:
        domain = np.ones(shape, dtype=bool)
    else:
        mask = valid_mask.astype(bool, copy=False)
        try:
            domain = np.broadcast_to(mask, shape)
        except ValueError:
            if len(shape) == 3 and mask.shape == shape[1:]:
                domain = np.broadcast_to(mask[np.newaxis, :, :], shape)
            else:
                raise ValueError(
                    'Valid mask cannot be broadcast to the feature shape.'
                ) from None
    domain_size = int(np.count_nonzero(domain))
    if domain_size == 0:
        raise ValueError('Valid mask contains no feature samples.')
    for name, values in arrays.items():
        ratio = float(np.count_nonzero(domain & ~np.isfinite(values)) / domain_size)
        if ratio > max_missing_ratio:
            raise ValueError(
                f"Feature {name!r} missing-value ratio {ratio:.3f} exceeds "
                f'configured maximum {max_missing_ratio:.3f}.'
            )

    if strategy == 'drop':
        complete = domain & np.logical_and.reduce(
            [np.isfinite(values) for values in arrays.values()]
        )
        return {
            name: np.where(complete, values, np.nan).astype(np.float32)
            for name, values in arrays.items()
        }

    handled: dict[str, np.ndarray] = {}
    statistic = np.mean if strategy == 'mean' else np.median
    for name, values in arrays.items():
        finite = domain & np.isfinite(values)
        if not np.any(finite):
            raise ValueError(f"Feature {name!r} contains no finite values to impute.")
        fill_value = float(statistic(values[finite]))
        handled[name] = np.where(
            domain,
            np.where(np.isfinite(values), values, fill_value),
            np.nan,
        ).astype(np.float32)
    return handled
