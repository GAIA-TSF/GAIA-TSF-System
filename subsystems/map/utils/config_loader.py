"""YAML configuration loading for MAP workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a MAP YAML document into a plain, test-friendly mapping."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('MAP configuration must contain a YAML mapping.')
    config['_config_path'] = str(config_path)
    LOGGER.info('Loaded MAP configuration from %s', config_path)
    return config
