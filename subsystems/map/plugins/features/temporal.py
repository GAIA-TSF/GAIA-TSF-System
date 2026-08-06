import logging

from subsystems.map.core.registry import register_feature


LOGGER = logging.getLogger(__name__)

"""
Temporal feature engineering

Transforms time series into supervised learning format
"""


@register_feature('temporal')
def temporal_features(data, config):
    """
    Converts time series into supervised learning format.

    Key parameters:
    - look_back: number of past timesteps used as input
    - horizon: prediction horizon

    Output:
        X → features
        y → target

    Works for:
    - LSTM (sequence format)
    - RF/XGB (flattened features)
    """
    LOGGER.info("Temporal feature plugin is deprecated; DAG supplies engineered features.")

    # Mock output
    raise NotImplementedError("MAP consumes DAG-engineered rasters through DatasetBuilder.")
