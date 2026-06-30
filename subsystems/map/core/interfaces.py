from abc import ABC, abstractmethod


class VariablePlugin(ABC):
    """
    Encapsulates ALL variable-specific logic.

    Why:
    - Each variable (slope, AMD, etc.) has different preprocessing,
      feature engineering, and allowed models.
    - This prevents scattering logic across the codebase.
    """

    name: str  # unique identifier (used in config)

    @abstractmethod
    def preprocess(self, data, config):
        """
        Perform variable-specific preprocessing.

        Examples:
        - Slope: may do nothing (InSAR already processed)
        - AMD: gap filling + smoothing

        Input:
            raw data (time series)
        Output:
            cleaned/preprocessed data
        """
        pass

    @abstractmethod
    def feature_pipeline(self) -> str:
        """
        Returns name of feature pipeline to use.

        This decouples:
        - WHAT variable is used
        - HOW features are computed

        Example:
            "temporal", "lagged", "temporal_with_gapfill"
        """
        pass

    @abstractmethod
    def allowed_models(self) -> list[str]:
        """
        Restricts which models are valid for this variable.

        Example:
            slope → ["lstm", "rf"]
            amd   → ["xgb", "rf"]

        Prevents invalid combinations.
        """
        pass


class ModelPlugin(ABC):
    """
    Unified interface for ALL models (LSTM, RF, XGB, etc.)

    Key design:
    - Pipelines do NOT care about model type
    - Everything follows fit() / predict()
    """

    def __init__(self, config):
        self.config = config  # hyperparameters, etc.

    @abstractmethod
    def fit(self, X, y): # noqa: N803
        """
        Train model.

        X:
            - sequences (LSTM)
            - tabular features (RF/XGB)
        """
        pass

    @abstractmethod
    def predict(self, X): # noqa: N803
        """
        Must return PredictionResult.

        Why:
        - Monitoring requires standardized output
        - Enables uncertainty propagation later
        """
        pass


class PredictionResult:
    """
    Standardized prediction container.

    Why:
    - Monitoring module expects consistent structure
    - Enables future extension (uncertainty, quantiles, etc.)
    """

    def __init__(self, y_pred, uncertainty=None):
        self.y_pred = y_pred
        self.uncertainty = uncertainty
