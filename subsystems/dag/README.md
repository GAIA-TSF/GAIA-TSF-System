# Data Aggregation (DAG) Sub-system 

The **Data Aggregation** sub-system serves as the critical
processing bridge that transforms data stored within the
Spatial Data Infrastructure (SDI) into structured inputs suitable for
machine learning analysis. Its primary function is to prepare data in
structures that are harmonised and ready for downstream consumption,
adhering to the principle that "garbage in, garbage out" dictates
model performance. The sub-system ingests multi-temporal satellite
image stacks (e.g., Sentinel-2) and co-located in-situ measurements
(e.g., pH, pore pressure) to produce model-ready feature tensors.

![Data Agregation Architecture](../../images/dag_subsystem.png)

## Key Capabilities

- Multi-temporal EO data ingestion (Sentinel-1, Sentinel-2)
- Exploratory Data Analysis (EDA): step important to select the following steps 
- Spatial harmonization (resampling to common grid)
- Feature engineering:
  - Slope stability: displacement → velocity → acceleration
  - AMD: spectral indices and AMD index
  - Meteorology: precipitation accumulations/extremes and temperature metrics
- Masking (AOI, water mask)
- Data preprocessing:
  - normalization (min-max, z-score)
  - missing value handling
  - outlier handling / log transform
- Validation and consistency checks


## Inputs

- **Slope Stability (KV1)**
  - Sentinel-1 LOS displacement time series
  - AOI mask

- **AMD (KV2)**
  - Sentinel-2 multispectral time series
  - AOI mask
  - TSF, clean water mask (optionally leak water mask) 

## Outputs

- graphs and maps 
- ML-ready features: spatiotemporal feature cubes
- Derived features:
  - velocity, acceleration (slope stability), etc. 
  - AMD index spectral features and its temporal derivatives 
- Ready for probabilistic anomaly detection (MAP subsystem)

## Architecture

**Workflow:** Raw EO Data -> Ingestion -> Preprocessing (Harmonization: spatial / temporal) -> Masking (AOI, water) -> Feature Engineering -> ML-ready dataset 

### Meteorological features

Meteorological processing is available through the `meteo_features` pipeline.
The repository configuration is ready for the synthetic project's daily CSV:

```bash
python3 subsystems/dag/run_pipeline.py \
  --pipeline meteo_features \
  --config subsystems/dag/config.yaml
```

The pipeline writes one dated multiband GeoTIFF per enabled feature and a
`metadata.json` file to `meteorology.results.output_dir`.

The default configuration reads daily observations from `inputs/meteodata.csv`
and broadcasts them over the configured TSF mask. Features are engineered on
the daily series and then sampled on `meteorology.inputs.insar` acquisition
dates, making every output band align with the InSAR temporal axis. A CSV must
contain `date`,
`precipitation`, `temperature_mean`, `temperature_min`, and `temperature_max`
columns. Separate dated GeoTIFF series are also supported through per-variable
`directory` and `filename_pattern` input mappings.


## Run the pipelines 

Run slope stability pipelines 

```bash
python3 subsystems/dag/run_pipeline.py  \
  --pipeline slope_eda \
  --config subsystems/dag/config.yaml

```

```bash
python3 subsystems/dag/run_pipeline.py \
  --pipeline slope_features \
  --config subsystems/dag/config.yaml
```

```bash
python3 subsystems/dag/run_pipeline.py \
  --pipeline meteo_features \
  --config subsystems/dag/config.yaml
```

```bash
python3 subsystems/dag/run_pipeline.py \
  --pipeline slope_temporal_features \
  --config subsystems/dag/config.yaml
```


## Testing 

```
PYTHONPATH=. pytest subsystems/dag/tests
```
