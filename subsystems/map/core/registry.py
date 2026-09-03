from __future__ import annotations

from typing import Any, Callable

# Global registries (simple and explicit)
VARIABLE_REGISTRY: dict[str, Any] = {}
MODEL_REGISTRY: dict[str, type] = {}
FEATURE_REGISTRY: dict[str, Callable[..., Any]] = {}
MONITORING_REGISTRY: dict[str, Any] = {}
EXPLAINABILITY_REGISTRY: dict[str, Any] = {}
OPERATION_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_variable(name: str) -> Callable[[type], type]:
    """Register a variable plugin and instantiate it immediately."""

    def decorator(cls: type) -> type:
        VARIABLE_REGISTRY[name] = cls()
        return cls

    return decorator


def register_model(name: str) -> Callable[[type], type]:
    """Register a model class for later instantiation."""

    def decorator(cls: type) -> type:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_feature(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a feature engineering function."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        FEATURE_REGISTRY[name] = fn
        return fn

    return decorator


def register_monitoring(name: str) -> Callable[[type], type]:
    """Register a monitoring plugin class."""

    def decorator(cls: type) -> type:
        MONITORING_REGISTRY[name] = cls
        return cls

    return decorator


def register_explainability(name: str) -> Callable[[type], type]:
    """Register an explainability plugin class."""

    def decorator(cls: type) -> type:
        EXPLAINABILITY_REGISTRY[name] = cls
        return cls

    return decorator


def register_operation(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a DAG operation callable."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        OPERATION_REGISTRY[name] = fn
        return fn

    return decorator
