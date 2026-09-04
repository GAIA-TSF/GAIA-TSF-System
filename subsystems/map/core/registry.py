"""Small explicit registries used to keep MAP pipelines plugin agnostic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar('T')
VARIABLE_REGISTRY: dict[str, Any] = {}
MODEL_REGISTRY: dict[str, type[Any]] = {}
FEATURE_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_variable(name: str) -> Callable[[type[T]], type[T]]:
    """Register a stateless variable plugin instance under ``name``."""

    def decorator(plugin_class: type[T]) -> type[T]:
        if name in VARIABLE_REGISTRY:
            raise ValueError(f'Variable plugin already registered: {name}')
        VARIABLE_REGISTRY[name] = plugin_class()
        return plugin_class

    return decorator


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Register a predictive model class under ``name``."""

    def decorator(plugin_class: type[T]) -> type[T]:
        if name in MODEL_REGISTRY:
            raise ValueError(f'Model plugin already registered: {name}')
        MODEL_REGISTRY[name] = plugin_class
        return plugin_class

    return decorator


def register_feature(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Register an optional feature plugin under ``name``."""

    def decorator(function: Callable[..., T]) -> Callable[..., T]:
        if name in FEATURE_REGISTRY:
            raise ValueError(f'Feature plugin already registered: {name}')
        FEATURE_REGISTRY[name] = function
        return function

    return decorator
