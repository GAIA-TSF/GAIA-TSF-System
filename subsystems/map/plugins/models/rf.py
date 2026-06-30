from sklearn.ensemble import RandomForestRegressor
from core.registry import register_model
from core.interfaces import ModelPlugin, PredictionResult


@register_model('rf')
class RFModel(ModelPlugin):
    def __init__(self, config):
        self.model = RandomForestRegressor(**config.rf_params)

    def fit(self, X, y): # noqa: N803 
        """
        Standard tabular training.
        """
        self.model.fit(X, y) # noqa: N803 

    def predict(self, X): # noqa: N803 
        y_pred = self.model.predict(X) # noqa: N803 

        # Optional uncertainty (variance across trees)
        uncertainty = None

        return PredictionResult(y_pred, uncertainty)
