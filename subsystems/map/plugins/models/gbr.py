"""
Gradient Boosting Regressor plugin (sklearn)

Replaces XGBoost with a stable, built-in alternative.
"""

from sklearn.ensemble import GradientBoostingRegressor
from core.registry import register_model


class PredictionResult:
    """
    Standard prediction container.

    Compatible with monitoring module.
    """

    def __init__(self, y_pred, uncertainty=None):
        self.y_pred = y_pred
        self.uncertainty = uncertainty


@register_model('gbr')
class GBRModel:
    """
    sklearn Gradient Boosting model.

    Suitable for:
    - tabular features
    - nonlinear relationships
    """

    def __init__(self, config):
        params = (
            config.gbr_params.__dict__
            if hasattr(config.gbr_params, '__dict__')
            else config.gbr_params
        )

        print(f'[Model] Initialize GBR with params: {params}')

        self.model = GradientBoostingRegressor(**params)

    def fit(self, X, y): # noqa: N803 
        """
        Train model on tabular features.
        """
        print(f'[Model] Training on X={X}, y={y}')
        # self.model.fit(X, y)

    def predict(self, X): # noqa: N803 
        """
        Generate predictions.
        """
        print(f'[Model] Predicting on X={X}')

        # y_pred = self.model.predict(X)
        y_pred = 'y_pred_mock'

        return PredictionResult(y_pred)
