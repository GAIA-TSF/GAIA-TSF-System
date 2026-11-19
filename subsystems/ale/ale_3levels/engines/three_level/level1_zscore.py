"""Shared L1 residual formula and frozen, pre-monitoring scale calibration."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from ale_3levels.engines.three_level.calibration import (
    SCHEMA_VERSION,
    compatible_schema,
)

FORMULA = 'z = (observed - forecast) / sigma'


def positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')
    return value


def residual_scale(residuals: np.ndarray) -> tuple[float, int]:
    """Sample residual standard deviation; never centre the scoring formula."""
    values = np.asarray(residuals, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError('Need at least two finite calibration residuals.')
    sigma = positive_finite(np.std(values, ddof=1), 'Calibration residual sigma')
    return sigma, int(values.size)


def level1_scores(
    residuals: np.ndarray,
    sigma: float,
    threshold: float = 3.0,
    direction: str = 'both',
    monitoring_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous L1: strict directional z exceedance, without persistence."""
    sigma = positive_finite(sigma, 'Residual sigma')
    threshold = positive_finite(threshold, 'Z-score threshold')
    if direction not in {'both', 'positive', 'negative'}:
        raise ValueError('L1 direction must be both, positive, or negative.')
    residuals = np.asarray(residuals, dtype=np.float64)
    z = np.where(np.isfinite(residuals), residuals / sigma, np.nan)
    evidence = (
        -z if direction == 'negative' else z if direction == 'positive' else np.abs(z)
    )
    flags = np.isfinite(z) & (evidence > threshold)
    if monitoring_mask is not None:
        flags &= monitoring_mask
    return z, flags


def frozen_residual_calibration(
    path: Path,
    residual_stack: np.ndarray,
    dates: tuple[str, ...],
    calibration_window: tuple[int, int],
    monitoring_start: int,
    mask: np.ndarray,
    *,
    unit: str,
    target_feature: str,
) -> dict[str, Any]:
    """Fit once from pre-monitoring calibration pixels; reject changed history.

    A supplied earlier artifact may be reused when this stack has no calibration
    history. Its cutoff must still precede monitoring and its unit/target match.
    Monitoring residuals are neither hashed nor used to fit the scale.
    """
    if residual_stack.shape != (len(dates), *mask.shape):
        raise ValueError(
            'L1 residual stack, dates and mask must share a grid/time axis.'
        )
    parsed_dates = tuple(date.fromisoformat(value) for value in dates)
    if any(right <= left for left, right in pairwise(parsed_dates)):
        raise ValueError('L1 acquisition dates must be unique and chronological.')
    if not 0 <= monitoring_start < len(dates):
        raise ValueError('L1 monitoring start is outside acquisition dates.')
    start, end = calibration_window
    if not 0 <= start < end <= len(dates):
        raise ValueError('L1 calibration window is outside acquisition dates.')
    end = min(end, monitoring_start)
    values = np.asarray(residual_stack[start:end, mask], dtype=np.float64)
    has_history = start < end and np.any(np.isfinite(values))
    fingerprint = None
    if has_history:
        digest = hashlib.sha256()
        digest.update(json.dumps(dates[start:end]).encode())
        digest.update(str(mask.shape).encode())
        digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
        digest.update(
            np.where(np.isfinite(values), values, np.nan).astype('<f8').tobytes()
        )
        fingerprint = digest.hexdigest()
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if (
            not compatible_schema(payload.get('schema_version'))
            or payload.get('method') != 'sample_std_ddof1_no_demeaning'
            or payload.get('unit') != unit
            or payload.get('target_feature') != target_feature
        ):
            raise ValueError(
                'Frozen L1 calibration has an incompatible method, unit or target.'
            )
        positive_finite(payload['sigma'], 'Frozen calibration sigma')
        if int(payload.get('sample_count', 0)) < 2:
            raise ValueError('Frozen L1 calibration must contain at least two samples.')
        if (
            not payload.get('last_date')
            or payload['last_date'] >= dates[monitoring_start]
        ):
            raise ValueError('Frozen L1 calibration must end before monitoring starts.')
        if fingerprint is not None and payload.get('input_fingerprint') != fingerprint:
            raise ValueError(
                'L1 calibration history/support changed. Use a new calibration_file '
                'or run directory for an explicitly recalibrated model.'
            )
        return payload
    if not has_history:
        raise ValueError(
            'No finite calibration residuals precede monitoring. Configure a prior '
            'calibration period or supply an existing frozen calibration_file.'
        )
    sigma, count = residual_scale(values)
    finite = values[np.isfinite(values)]
    payload = {
        'schema_version': SCHEMA_VERSION,
        'method': 'sample_std_ddof1_no_demeaning',
        'formula': FORMULA,
        'population': 'pooled_pixel_residuals_on_static_tsf_mask',
        'unit': unit,
        'target_feature': target_feature,
        'sigma': sigma,
        'sample_count': count,
        'mean_residual_diagnostic': float(np.mean(finite)),
        'first_date': dates[start],
        'last_date': dates[end - 1],
        'calibration_dates': list(dates[start:end]),
        'input_fingerprint': fingerprint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing another run's calibration.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return payload
