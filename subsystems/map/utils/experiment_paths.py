"""Experiment-scoped MAP output-path helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_EXPERIMENT_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


def experiment_model_directory(output_root: Path, config: dict[str, Any]) -> Path:
    """Return the isolated model-artifact directory for one MAP experiment.

    The configured experiment name is intentionally used as a single path
    component. This keeps baseline and tRF model artifacts separate and makes
    every experiment directly identifiable from its output directory.

    Args:
        output_root: Configured MAP results root.
        config: Full MAP configuration containing ``experiment.name``.

    Returns:
        ``<output_root>/models/<experiment.name>``.

    Raises:
        ValueError: If the experiment name is missing or unsafe as a directory.
    """
    experiment = config.get('experiment')
    name = experiment.get('name') if isinstance(experiment, dict) else None
    if not isinstance(name, str) or not _EXPERIMENT_NAME.fullmatch(name):
        raise ValueError(
            'experiment.name must contain only letters, digits, dots, hyphens, '
            'and underscores, and must start with a letter or digit.',
        )
    return output_root / 'models' / name
