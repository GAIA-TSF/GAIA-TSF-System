"""Plugin package initialization for the MAP subsystem."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def load_plugins() -> None:
    """Import plugin modules so their decorators register them."""
    package_root = Path(__file__).resolve().parent
    for module_path in package_root.rglob("*.py"):
        if module_path.name.startswith("_") or module_path.name == "__init__.py":
            continue
        relative = module_path.relative_to(package_root)
        module_name = ".".join(relative.with_suffix("").parts)
        importlib.import_module(f"plugins.{module_name}")


load_plugins()
