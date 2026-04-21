
from core.registry import register_feature

"""
Temporal feature engineering

Transforms time series into supervised learning format
"""

@register_feature("temporal")
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
    print(f"[Features] Temporal features (look_back={config.look_back}, horizon={config.horizon})") 

    # Mock output
    X = "X_features"
    y = "y_target"

    # return create_temporal_features(
    #     data,
    #     look_back=config.look_back,
    #     horizon=config.horizon
    # )
    return X, y 
