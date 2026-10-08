# MAP subsystem architecture

The MAP subsystem is the predictive analytics core of GAIA-TSF. It consumes engineered features from the DAG subsystem and produces baseline models, predictions, residuals, anomaly summaries, and explainability artifacts without embedding variable-specific logic in the core pipeline.

## Architecture overview

- Learning pipeline: executes the configured `pipelines.learning.dag`, builds pixel-time samples from configured feature rasters, filters training rows through `StablePixelSelector`, trains a registered `PredictiveModel`, validates it, persists artifacts, and records the model version.
- Inference pipeline: executes the configured `pipelines.inference.dag`, loads a trained model, rebuilds the same configured dataset, scores all monitored pixels in the inference split, and stores predictions and residuals independently.
- Monitoring pipeline: evaluates residuals through configured plugins such as residual thresholding, z-score, CUSUM, BOCD, and regime detection, then writes combined anomaly score and binary anomaly arrays.
- Explainability pipeline: runs configured explainability plugins such as SHAP, LIME, and DiCE and stores method summaries under `results/explainability/`.

## Key components

- `config.yaml`: active variable, feature pipeline, model, training split, stable-pixel threshold, monitoring methods, and explainability methods.
- `core/`: shared interfaces, plugin and operation registries, DAG executor, experiment manager, and persistent model registry.
- `dataset/`: feature loading, dataset construction, temporal splitting, and window utilities.
- `plugins/variables/`: variable-specific preprocessing and allowed-model declarations.
- `plugins/selection/`: replaceable baseline-training pixel selectors.
- `plugins/models/`: interchangeable predictive models implementing `train()`, `predict()`, `save()`, and `load()`.
- `plugins/monitoring/`: interchangeable residual/anomaly monitoring methods.
- `plugins/explainability/`: explainability plugin interfaces and registered placeholders.
- `pipelines/`: DAG-backed learning/inference wrappers, operation adapters, monitoring, and explainability orchestration.

## Configurable DAGs

Learning and inference execution order is configured in `config.yaml` under `pipelines`. Each node names an operation registered in `core.registry.OPERATION_REGISTRY` and declares dependencies through `inputs`.

Example:

```yaml
pipelines:
  learning:
    dag:
      nodes:
        load:
          op: tensor_loader
          inputs: [input]
        features:
          op: feature_engineering
          inputs: [load]
        stable:
          op: stable_pixel_selection
          inputs: [features]
        train:
          op: trainer
          inputs: [stable]
        validate:
          op: validation
          inputs: [train]
      output: validate
```

Built-in operations include `tensor_loader`, `splitter`, `windowing`, `feature_engineering`, `stable_pixel_selection`, `trainer`, `validation`, `predictor`, `residual_analysis`, `trend_detection`, `anomaly_detection`, and `risk_scoring`.

## Usage

Training:

```bash
python subsystems/map/run_learning.py --config subsystems/map/config.yaml
```

Inference:

```bash
python subsystems/map/run_inference.py --config subsystems/map/config.yaml
```

Monitoring:

```bash
python -c "import sys, numpy as np; sys.path.insert(0, 'subsystems/map'); from utils.config_loader import load_config; from pipelines.monitoring_pipeline import run_monitoring_pipeline; config = load_config('subsystems/map/config.yaml'); residuals = np.load('results/residuals/residual_slope.npy'); run_monitoring_pipeline(config, residuals)"
```

Explainability:

```bash
python -c "import sys, numpy as np; sys.path.insert(0, 'subsystems/map'); from utils.config_loader import load_config; from core.registry import MODEL_REGISTRY; import plugins.models.rf; from pipelines.explainability_pipeline import run_explainability_pipeline; config = load_config('subsystems/map/config.yaml'); model = MODEL_REGISTRY['rf'].load('results/models/slope_rf.joblib'); X = np.load('results/predictions/prediction_slope.npy').reshape(-1, 1); run_explainability_pipeline(config, model, X)"
```

## Extensibility

Future variables and monitoring methods can be added by implementing a new plugin class and registering it in the appropriate plugin package. The dataset builder and orchestration pipelines remain unchanged because feature names, variable selection, model selection, thresholds, and active monitoring/explainability methods are configuration-driven.

## Outputs

- `results/models/`: serialized model, metrics, and versioned model registry metadata.
- `results/predictions/`: prediction arrays for monitored pixels.
- `results/residuals/`: residual arrays and residual statistics.
- `results/anomalies/`: combined anomaly score, binary anomaly arrays, and anomaly summary JSON.
- `results/explainability/`: explainability plugin summaries grouped by method.
