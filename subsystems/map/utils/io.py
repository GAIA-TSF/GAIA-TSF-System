"""Deprecated module retained to avoid breaking prototype imports."""

from __future__ import annotations

from pathlib import Path

from subsystems.map.plugins.models.predictive_model import PredictiveModel


def load_model(path: str | Path, model_class: type[PredictiveModel]) -> PredictiveModel:
    """Load a MAP model with the explicit plugin class required for safety."""
    return model_class.load(Path(path))
