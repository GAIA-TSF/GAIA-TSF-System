from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar('T')


class PluginRegistry:
    """Registry for creating plugins by name."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], object]] = {}

    def register(self, name: str, factory: Callable[[], T]) -> None:
        """Register a plugin factory.

        Args:
            name: Unique plugin name.
            factory: Callable that creates a plugin instance.

        Raises:
            ValueError: If a plugin with the same name already exists.
        """
        if name in self._factories:
            raise ValueError(f'Plugin already registered: {name}')
        self._factories[name] = factory

    def create(self, name: str) -> object:
        """Create a registered plugin.

        Args:
            name: Registered plugin name.

        Returns:
            Plugin instance.

        Raises:
            KeyError: If the plugin is not registered.
        """
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f'Plugin is not registered: {name}') from exc
        return factory()

    def names(self) -> list[str]:
        """Return registered plugin names."""
        return sorted(self._factories)


PLUGIN_REGISTRY = PluginRegistry()
