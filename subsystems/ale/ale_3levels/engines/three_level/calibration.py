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
