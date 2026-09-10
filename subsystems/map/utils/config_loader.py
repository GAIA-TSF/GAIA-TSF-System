"""YAML config loader for the MAP subsystem."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("map.config_loader")


class Config:
    """
    Simple wrapper to allow dot access:
        config.variable instead of config["variable"]
    """

    def __init__(self, config_dict: dict[str, Any]) -> None:
        for key, value in config_dict.items():
            # Recursively convert nested dictionaries
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __repr__(self) -> str:
        return str(self.__dict__)


def load_config(path: str | Path = 'config.yaml') -> Config:
    """
    Load YAML file and return Config object
    """
    logger.info("Loading config from %s", path)

    with Path(path).open('r', encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    return Config(config_dict)
