from core.registry import register_feature


@register_feature('lagged')
def lagged_features(data, config):
    """
    Simple lag-based representation:

    X_t = [y_{t-1}, y_{t-2}, ..., y_{t-k}]

    Often very effective for:
    - RF
    - XGBoost

    Less suitable for:
    - long temporal dependencies
    """
    # return create_lagged_features(data, lags=config.lags)
