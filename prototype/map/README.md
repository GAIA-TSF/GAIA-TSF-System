# Machine Learning Predictive Analytics (MAP) Sub-system

Trend detection, anomaly analysis, and dynamic risk scoring.

The **Machine Learning Predictive Analytics** sub-system functions as
the core intelligence engine of the GAIA-TSF architecture, responsible
for transforming historical and current monitoring data into
actionable predictive insights (trend, anomaly, and risk scoring). The
sub-system is designed to execute comprehensive modeling workflows
that include data ingestion, feature engineering, model training,
inference, and output results upload.

![ML Predictive Analytics](../../images/map_subsystem.png)

### Multi-Variable Time Series Monitoring Framework
This subsystem implements a modular machine learning framework for monitoring geophysical processes from time series data. It is designed to support multiple variables (e.g., slope stability, AMD) and multiple model types (e.g., LSTM, Random Forest, XGBoost) within a unified, reproducible pipeline.

### Key Features
* **Plugin Architecture**
  Variables, feature engineering pipelines, and models are implemented as independent plugins. This enables flexible combinations without modifying core code.

* **Multi-Variable Support**
  * **Slope Stability** (InSAR displacement time series)
    * Models: LSTM, Random Forest
    * Features: engineered temporal features

  * **AMD** (Sentinel-2 AMD index time series)`
    * Models: XGBoost / LightGBM, Random Forest
    * Features: gap filling, noise filtering, temporal features

* **Unified Pipelines**
`
  * `learning_pipeline`: training and experiment registration
  * `inference_pipeline`: prediction, residual computation, and monitoring

* **Physics-Informed Monitoring Layer**
  Predictions are transformed into risk signals using methods such as:

  * CUSUM (persistent acceleration detection)
  * Bayesian online change point detection
  * Regime classification (acceleration, deceleration, oscillation)


### Architecture Overview

```
map/
├── core/              # interfaces + registry 
├── dataset/           # data loading, windowing, splitting
├── learning/          # training logic 
├── inference/         # prediction logic 
├── monitoring/        # physics-informed layer
├── pipelines/         # orchestration 
├── plugins/           # extensibility 
│   ├── features/
│   ├── models/
│   └── variables/
├── utils/             # config, IO, helpers
├── tests/
```

### Learning Pipeline: 
config
  ↓
dataset →  variable plugin
  ↓
feature pipeline
  ↓
learning.trainer
  ↓
model
  ↓
save

### Inference Pipeline 

config
  ↓
dataset → variables 
  ↓
feature pipeline
  ↓
inference.predictor
  ↓
residuals
  ↓
monitoring.runner


### Usage 

```yaml
variable: amd
model: xgb

look_back: 12
horizon: 1
```

```bash
python run_learning.py --config config.yaml
python run_inference.py --config config.yaml
```


### Test

```bash 
TODO: 
```

