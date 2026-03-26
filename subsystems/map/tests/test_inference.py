import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from subsystems.map.learning.lstm_model import LstmModel
from subsystems.map.inference.predictor import Predictor


class TestInferenceModule:
    """
    Tests requirement ML_R_03:
    Inference module applies trained model to new data
    and generates predictions and anomaly scores.
    """

    def make_predictor(self, horizon):
        model = LstmModel(
            input_size=1,
            hidden_size=8,
            num_layers=1,
            output_size=1,
            horizon=horizon,
            mode='forecasting',
        )

        predictor = Predictor(
            model=model,
            device=torch.device('cpu'),
            look_back=5,
            horizon=horizon,
            mc_samples=1,
            sigma_threshold=2.0,
            warmup_factor=2,
            calibration_fraction=0.4,
            persistence=3,
            use_model_uncertainty=True,
        )

        return predictor

    def test_model_prediction(self):
        predictor = self.make_predictor(horizon=3)

        series = np.random.randn(50)

        mean_pred, std_pred = predictor.predict_series(series)

        assert len(mean_pred) == len(series)
        assert len(std_pred) == len(series)

    def test_residual_computation(self):
        predictor = self.make_predictor(horizon=2)

        series = np.random.randn(50)

        mean_pred, std_pred = predictor.predict_series(series)

        residuals = predictor.compute_residuals(series, mean_pred)

        assert len(residuals) == len(series)

        # ignore warmup region where prediction is undefined
        # assert np.isfinite(residuals[predictor._look_back:]).all()
        valid = residuals[predictor.look_back : -predictor.horizon + 1]
        assert np.isfinite(valid).all()

    def test_anomaly_detection(self):
        predictor = self.make_predictor(horizon=2)

        series = np.random.randn(50)

        mean_pred, std_pred = predictor.predict_series(series)

        residuals = predictor.compute_residuals(series, mean_pred)

        score, threshold, mask = predictor.detect_anomaly(
            residuals,
            std_pred,
        )

        assert len(score) == len(series)
        assert len(mask) == len(series)
