"""Experiment-scoped MAP output-path helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_EXPERIMENT_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


def experiment_directory(config: dict[str, Any], config_path: Path) -> Path:
    """Resolve the one configured TSF experiment directory.

    MAP follows the DAG scenario layout: engineered products live below
    ``<experiment_dir>/results`` and static spatial inputs below
    ``<experiment_dir>/static``. Keeping this root in one configuration key
    prevents feature, mask, output, and point-layer paths from drifting apart.
    """
    value = config.get('experiment_dir')
    if not isinstance(value, str) or not value.strip():
        raise KeyError('MAP configuration requires a non-empty experiment_dir.')
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def results_directory(config: dict[str, Any], config_path: Path) -> Path:
    """Return the standard DAG/MAP results directory for the experiment."""
    return experiment_directory(config, config_path) / 'results'


def static_file_path(
    config: dict[str, Any],
    config_path: Path,
    filename: object,
) -> Path:
    """Return a static experiment file from a filename relative to ``static``."""
    path = Path(str(filename)).expanduser()
    return (
        path
        if path.is_absolute()
        else experiment_directory(config, config_path) / 'static' / path
    )


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
