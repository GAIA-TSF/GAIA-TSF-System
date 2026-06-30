from core.registry import VARIABLE_REGISTRY, FEATURE_REGISTRY
from dataset.loader import load_new_data
from utils.io import load_model
from monitoring.monitoring import run_monitoring

"""
Inference pipeline

Runs:
data → prediction → residuals → monitoring
"""


def run_inference(config):
    """
    End-to-end inference + monitoring.

    Pipeline:
        data → preprocess → features → prediction → residuals → monitoring
    """

    print('\n=== INFERENCE PIPELINE ===')

    variable = VARIABLE_REGISTRY[config.variable]

    # 1. Load new/unseen data
    data = load_new_data(config)

    # 2. Apply same preprocessing as training
    data = variable.preprocess(data, config)

    # 3. Same feature pipeline (CRITICAL for consistency)
    feature_fn = FEATURE_REGISTRY[variable.feature_pipeline()]
    X, y = feature_fn(data, config) # noqa: N803

    # 4. Load trained model
    model = load_model(config)

    # 5. Predict
    result = model.predict(X) # noqa: N803 

    # 6. Residuals = observed - predicted
    residuals = y - result.y_pred

    # 7. Monitoring module
    monitoring_output = run_monitoring(residuals, config)

    return {
        'prediction': result,
        'residuals': residuals,
        'monitoring': monitoring_output,
    }
