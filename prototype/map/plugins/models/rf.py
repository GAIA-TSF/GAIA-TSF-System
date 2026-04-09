
from sklearn.ensemble import RandomForestRegressor
from core.registry import register_model
from core.interfaces import ModelPlugin, PredictionResult


@register_model("rf")
class RFModel(ModelPlugin):

    def __init__(self, config):
        self.model = RandomForestRegressor(**config.rf_params)

    def fit(self, X, y):
        """
        Standard tabular training.
        """
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)

        # Optional uncertainty (variance across trees)
        uncertainty = None

        return PredictionResult(y_pred, uncertainty)
    