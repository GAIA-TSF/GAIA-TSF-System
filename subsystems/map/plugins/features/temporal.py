from core.registry import register_feature

import logging
from typing import Any

"""
Temporal feature engineering

Transforms time series into supervised learning format
"""

logger = logging.getLogger("map.features.temporal")


@register_feature('temporal')
def temporal_features(data: Any, config: Any) -> tuple[str, str]:
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
    logger.info(
        "Temporal features configured with look_back=%s horizon=%s",
        getattr(config, "look_back", None),
        getattr(config, "horizon", None),
    )

    # Mock output
    X = 'X_features'  # noqa: N806
    y = 'y_target'

    # return create_temporal_features(
    #     data,
    #     look_back=config.look_back,
    #     horizon=config.horizon
    # )
    return X, y  # noqa: N803
