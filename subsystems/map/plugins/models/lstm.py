from core.registry import register_model
from core.interfaces import ModelPlugin, PredictionResult


@register_model('lstm')
class LSTMModel(ModelPlugin):
    def __init__(self, config):
        super().__init__(config)

        # Build architecture from config
        # self.model = build_lstm(config)

    def fit(self, X, y):  # noqa: N803
        """
        Training loop:
        - batching
        - optimizer
        - early stopping

        Encapsulated to keep pipeline clean.
        """
        # train_lstm(self.model, X, y, self.config)

    def predict(self, X):  # noqa: N803
        """
        Returns standardized output.

        Could later include:
        - MC dropout uncertainty
        - ensembles
        """
        y_pred = self.model.predict(X)  # noqa: N803

        return PredictionResult(y_pred)
