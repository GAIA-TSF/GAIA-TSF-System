"""Persisted engine-B state between runs (B2.6).

ALE re-reads the full, immutable MAP history on every run, so engine B
recomputes its accumulators causally. The persisted state carries what must
stay consistent across runs and cannot be recomputed from the MAP files:

* the model identity and calibration fingerprints the accumulators were
  built with;
* the reset history (dates and reasons): CUSUM accumulators restart from
  zero after a recalibration, after a model-version change, and
  (``state.reset_after_l3``) after an L3 alert;
* the last processed date and the TSF-wide and per-pixel CUSUM accumulators
  there, to prove that a later run reproduces every issued value.

Reset rules when a previous state is given:

* same identity and calibration: earlier resets are applied again; the
  recomputed accumulators at the previous last date must match the stored
  ones (otherwise :class:`StateMismatchError`: the history changed);
* different ``model_version`` (or model/lead): reset at the first new date;
* different calibration (fingerprint or method): reset at the first new date.

After a recalibration or model change the output recomputes the history with
the new calibration/model; the values issued before remain those of the
earlier run's output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STATE_VERSION = '1.0'
RESET_REASONS = ('recalibration', 'model_version_change', 'l3_event')
# MAP outputs are immutable; a tolerance only absorbs float noise from a
# re-run of the same MAP model (repeated random-forest runs differ ~1e-14).
CONTINUITY_RTOL = 1e-9
CONTINUITY_ATOL = 1e-12


class StateMismatchError(ValueError):
    """The persisted state does not match the recomputed history."""


@dataclass(frozen=True)
class EngineState:
    """Engine-B state after a run (see module docstring)."""

    identity: dict[str, Any] | None
    calibration: dict[str, dict[str, Any]]
    last_date: str
    resets: tuple[dict[str, str], ...]
    tsf_cusum: tuple[float, float]
    pixel_cusum_pos: np.ndarray
    pixel_cusum_neg: np.ndarray
    tsf_version: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


def calibration_fingerprints(
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """What identifies a frozen calibration: method and input fingerprint per level."""
    return {
        level: {
            'method': payload.get('method'),
            'input_fingerprint': payload.get('input_fingerprint'),
        }
        for level, payload in sorted(calibrations.items())
    }


def plan_resets(
    previous: EngineState | None,
    identity: dict[str, Any] | None,
    calibration: dict[str, dict[str, Any]],
    dates: tuple[str, ...],
) -> list[dict[str, str]]:
    """Resets to apply in this run, from the previous state.

    Returns ``[{'date': ..., 'reason': ...}, ...]``; reset dates are
    acquisitions at which the accumulators restart from zero.
    """
    if previous is None:
        return []
    known = set(dates)
    carried = [dict(reset) for reset in previous.resets if reset['date'] in known]
    later = [day for day in dates if day > previous.last_date]
    if not later:
        return carried
    first_new = later[0]
    reasons = []
    if previous.identity != identity:
        reasons.append('model_version_change')
    if previous.calibration != calibration:
        reasons.append('recalibration')
    return carried + [{'date': first_new, 'reason': reason} for reason in reasons]


def check_continuity(
    previous: EngineState | None,
    resets: list[dict[str, str]],
    tsf_cusum: tuple[np.ndarray, np.ndarray],
    pixel_cusum: tuple[np.ndarray, np.ndarray],
    dates: tuple[str, ...],
) -> None:
    """The recomputed accumulators at the previous last date match the state.

    Skipped when this run resets for a recalibration or a model-version change:
    the history is then recomputed with the new calibration or model, so it
    differs by design. The values issued earlier stay in the earlier output;
    the accumulators restart from zero at the first new date.
    """
    if previous is None:
        return
    if any(
        reset['date'] > previous.last_date
        and reset['reason'] in ('recalibration', 'model_version_change')
        for reset in resets
    ):
        return
    if previous.last_date not in dates:
        raise StateMismatchError(
            f'The previous state ends on {previous.last_date}, which is not an '
            'acquisition of this input; the MAP history changed.'
        )
    index = dates.index(previous.last_date)
    recomputed = (
        np.array([tsf_cusum[0][index], tsf_cusum[1][index]], dtype=np.float64),
        pixel_cusum[0][index],
        pixel_cusum[1][index],
    )
    stored = (
        np.array(previous.tsf_cusum, dtype=np.float64),
        previous.pixel_cusum_pos,
        previous.pixel_cusum_neg,
    )
    for name, new, old in zip(('tsf', 'pixel_pos', 'pixel_neg'), recomputed, stored):
        if new.shape != old.shape or not np.allclose(
            new, old, rtol=CONTINUITY_RTOL, atol=CONTINUITY_ATOL, equal_nan=True
        ):
            raise StateMismatchError(
                f'Recomputed {name} CUSUM on {previous.last_date} differs from the '
                'persisted state: issued results would change. The MAP history or '
                'the engine configuration changed; start a new state.'
            )


def save_state(state: EngineState, directory: Path) -> Path:
    """Write ``state.json`` and ``state.npz`` (deterministic)."""
    from ale_3levels.outputs import write_json, write_npz

    directory = Path(directory)
    write_npz(
        directory / 'state.npz',
        {
            'pixel_cusum_pos': state.pixel_cusum_pos,
            'pixel_cusum_neg': state.pixel_cusum_neg,
        },
    )
    write_json(
        directory / 'state.json',
        {
            'state_version': STATE_VERSION,
            'identity': state.identity,
            'calibration': state.calibration,
            'last_date': state.last_date,
            'resets': list(state.resets),
            'tsf_cusum': [_number(value) for value in state.tsf_cusum],
            'tsf_version': state.tsf_version,
            'attrs': state.attrs,
        },
    )
    return directory


def load_state(directory: Path) -> EngineState:
    directory = Path(directory)
    payload = json.loads((directory / 'state.json').read_text(encoding='utf-8'))
    if payload.get('state_version') != STATE_VERSION:
        raise StateMismatchError(f'Unsupported state version in {directory}.')
    with np.load(directory / 'state.npz', allow_pickle=False) as archive:
        positive, negative = archive['pixel_cusum_pos'], archive['pixel_cusum_neg']
    return EngineState(
        identity=payload['identity'],
        calibration=payload['calibration'],
        last_date=payload['last_date'],
        resets=tuple(payload['resets']),
        tsf_cusum=tuple(
            np.nan if v is None else float(v) for v in payload['tsf_cusum']
        ),
        pixel_cusum_pos=positive,
        pixel_cusum_neg=negative,
        tsf_version=payload.get('tsf_version'),
        attrs=payload.get('attrs') or {},
    )


def _number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None
