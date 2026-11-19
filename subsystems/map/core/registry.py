"""Registry of the maintained MAP forecasting models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar('T')
MODEL_REGISTRY: dict[str, type[Any]] = {}


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Register a predictive model class under ``name``."""

    def decorator(plugin_class: type[T]) -> type[T]:
        if name in MODEL_REGISTRY:
            raise ValueError(f'Model plugin already registered: {name}')
        MODEL_REGISTRY[name] = plugin_class
        return plugin_class

    return decorator
