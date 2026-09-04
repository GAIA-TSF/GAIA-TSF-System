"""Temporal-feature Random Forest predictive-model plugin."""

from __future__ import annotations

from typing import Any

from subsystems.map.core.registry import register_model
from subsystems.map.plugins.models.rf import RFModel


@register_model('trf')
class TemporalRandomForestModel(RFModel):
    """Random Forest ML model for DAG-provided temporal feature vectors.

    Each row represents one pixel at one acquisition and its columns are the
    temporal features selected in the MAP dataset configuration.  The model
    deliberately performs no feature engineering: temporal lags, trends and
    other transformations remain the responsibility of the DAG subsystem.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the forest using the configured sklearn parameters."""
        super().__init__(config)
