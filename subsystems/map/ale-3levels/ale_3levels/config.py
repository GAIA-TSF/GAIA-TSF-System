"""ALE configuration loading and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ENGINE_NAMES = ('fixed_gates', 'three_level')
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'ale.yaml'


def _read_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f'{path} must contain a YAML mapping.')
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load ``ale.yaml`` and the engine files it references (relative to it)."""
    path = Path(path).resolve()
    config = _read_yaml(path)
    engines = config.get('engines') or {}
    selected = engines.get('selected', 'both')
    if selected not in (*ENGINE_NAMES, 'both'):
        raise ValueError(f'engines.selected must be one of {ENGINE_NAMES} or both.')
    config['engine_configs'] = {
        name: _read_yaml(path.parent / str(engines.get(name, f'{name}.yaml')))
        for name in ENGINE_NAMES
    }
    config['_config_path'] = str(path)
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 of a configuration mapping (key order independent)."""
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
