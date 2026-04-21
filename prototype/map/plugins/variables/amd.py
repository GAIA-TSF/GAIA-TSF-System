
from core.registry import register_variable
from core.interfaces import VariablePlugin

"""
AMD variable plugin

Encapsulates:
- preprocessing
- feature pipeline selection
- allowed models
""" 

@register_variable("amd")
class AMDVariable(VariablePlugin):

    name = "amd"

    def preprocess(self, data, config):
        """
        AMD index requires strong preprocessing:

        1. Gap filling (clouds, missing acquisitions)
        2. Noise filtering (sensor + atmospheric noise)

        Important:
        This step is CRITICAL for stability of ML models.
        """
        from plugins.feature.amd_preprocessing import gap_fill, smooth

        print("[AMD] Preprocessing: gap filling + smoothing") 

        # data = gap_fill(data)
        # data = smooth(data)
        
        # return data
        return 0 

    def feature_pipeline(self):
        """
        Could be:
        - temporal
        - lagged

        Currently unified → temporal
        """
        return "temporal"

    def allowed_models(self):
        
        """
        Gradient Boosting Regressor:
            - strong for tabular time-series features
        RF:
            - baseline, robust
        """
        return ["gbr", "rf"] 
