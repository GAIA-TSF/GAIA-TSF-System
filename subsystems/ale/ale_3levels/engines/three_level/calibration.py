"""Frozen-calibration schema of engine B (moved from MAP ``utils/contract.py``).

The prototype's calibration artifacts (``l1/l2/l3_calibration.json``) record
``schema_version``; artifacts written before the contract settled on ``1.0``
remain valid. The calibration functions themselves still live in their level
modules (``level1_zscore``, ``level2_cusum``, ``level3_regime``) during the
pure move (B1); B2 consolidates them here.
"""

from __future__ import annotations

from typing import Any

CONTRACT_VERSION = '1.0'
SCHEMA_VERSION = '1.0'
COMPATIBLE_SCHEMA_VERSIONS = frozenset({'1.0', '1.0.0'})

# Training provenance of the MAP model (run_mode), as recorded by the prototype.
OPERATIONAL = 'operational'
RETROSPECTIVE = 'retrospective'
UNKNOWN = 'unknown'


def compatible_schema(value: Any) -> bool:
    """True for a persisted calibration artifact written under schema 1.0."""
    return value in COMPATIBLE_SCHEMA_VERSIONS


class CalibrationError(ValueError):
    """A frozen engine-B calibration cannot be used."""


class CalibrationMismatchError(CalibrationError):
    """A frozen calibration belongs to another model, version or lead (B2.3)."""


class MissingCalibrationError(CalibrationError):
    """A required frozen calibration file does not exist."""


IDENTITY_KEYS = ('model_id', 'model_version', 'lead')


def check_identity(
    payload: dict[str, Any], identity: dict[str, Any] | None, level: str
) -> None:
    """Refuse residuals whose model identity differs from the frozen calibration.

    A calibration fitted on one model's residuals says nothing about another
    model's (or another version's, or another lead's) residual scale.
    """
    if identity is None:
        return
    frozen = payload.get('identity')
    if frozen is None:
        raise CalibrationMismatchError(
            f'Frozen {level} calibration records no model identity; it cannot be '
            f'checked against residuals of {_describe(identity)}. Recalibrate.'
        )
    if {key: frozen.get(key) for key in IDENTITY_KEYS} != {
        key: identity.get(key) for key in IDENTITY_KEYS
    }:
        raise CalibrationMismatchError(
            f'Frozen {level} calibration was fitted for {_describe(frozen)}, but the '
            f'residuals come from {_describe(identity)}. Refusing to score; run '
            "'calibrate' for this model version or use its own calibration."
        )


def _describe(identity: dict[str, Any]) -> str:
    return (
        f'model {identity.get("model_id")!r} version {identity.get("model_version")!r} '
        f'lead {identity.get("lead")!r}'
    )
