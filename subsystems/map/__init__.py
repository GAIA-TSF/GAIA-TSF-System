from map.dataset import DatasetModule
from map.learning import LearningModule
from map.inference import InferenceModule
from map.xai import ExplainabilityModule


class MachineLearningAnomalyPrediction:
    """Machine Learning Anomaly Prediction (MAP) subsystem.

    Generic ML subsystem providing dataset handling, model
    training, inference, and explainable AI capabilities.
    """

    id = 'MAP'

    def __init__(self):
        self.dataset = DatasetModule()
        self.learning = LearningModule()
        self.inference = InferenceModule()
        self.xai = ExplainabilityModule()
