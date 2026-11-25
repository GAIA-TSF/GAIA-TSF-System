"""ALE configuration loading and hashing.

A configuration file may name a ``base`` file (relative to itself); its
mappings are merged recursively, the derived file winning. Relative file paths
(``input.residuals``, ``input.mask_file``, ``spatial.sectors_file``,
``engines.fixed_gates``, ``engines.three_level``) expand
``~`` and resolve against the directory of the file that sets them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ENGINE_NAMES = ('fixed_gates', 'three_level')
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'ale.yaml'
PATH_KEYS = (
    ('input', 'residuals'),
    ('input', 'mask_file'),
    ('spatial', 'sectors_file'),
    ('engines', 'fixed_gates'),
    ('engines', 'three_level'),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f'{path} must contain a YAML mapping.')
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(config: dict[str, Any], directory: Path) -> dict[str, Any]:
    for section, key in PATH_KEYS:
        value = (config.get(section) or {}).get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        config[section][key] = str(
            path if path.is_absolute() else (directory / path).resolve()
        )
    return config


def _load_layered(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f'Circular configuration base: {path}')
    config = _resolve_paths(_read_yaml(path), path.parent)
    base = config.pop('base', None)
    if base is None:
        return config
    base_path = (path.parent / str(base)).resolve()
    return _merge(_load_layered(base_path, (*seen, path)), config)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load an ALE configuration and the engine files it references."""
    path = Path(path).resolve()
    config = _load_layered(path)
    engines = config.get('engines') or {}
    selected = engines.get('selected', 'both')
    if selected not in (*ENGINE_NAMES, 'both'):
        raise ValueError(f'engines.selected must be one of {ENGINE_NAMES} or both.')
    # Engine files resolve against the file that names them (see PATH_KEYS);
    # unnamed ones default to the directory of the loaded file.
    config['engine_configs'] = {
        name: _read_yaml(Path(engines.get(name) or path.parent / f'{name}.yaml'))
        for name in ENGINE_NAMES
    }
    config['_config_path'] = str(path)
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 of a configuration mapping (key order independent)."""
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
