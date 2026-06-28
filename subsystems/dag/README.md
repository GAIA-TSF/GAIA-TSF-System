# Data Aggregation (DAG) Sub-system 

The **Data Aggregation** sub-system serves as the critical
processing bridge that transforms harmonised data stored within the
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
- Spatial harmonization (resampling to common grid)
- Feature engineering:
  - Slope stability: displacement → velocity → acceleration
  - AMD: spectral indices and AMD index
- Masking (AOI, water mask)
- Data preprocessing:
  - normalization (min-max, z-score)
  - missing value handling
  - outlier handling / log transform
- Validation and consistency checks
- Tensorization into ML-ready formats `(T, H, W, C)`

## Inputs

- **Slope Stability (KV1)**
  - Sentinel-1 LOS displacement time series
  - AOI polygon

- **AMD (KV2)**
  - Sentinel-2 multispectral time series
  - AOI polygon
  - Water mask

## Outputs

- ML-ready tensors:
  - `(T, H, W, C)` spatiotemporal feature cubes
- Derived features:
  - velocity, acceleration (slope stability)
  - AMD index and spectral features
- Ready for probabilistic anomaly detection (MAP subsystem)

## Architecture

See: `./docs/coding_notes.md` 

**Workflow:** Raw EO Data -> Ingestion -> Harmonization (spatial / temporal) -> Masking (AOI, water) -> Feature Engineering -> Preprocessing -> Validation -> Tensorization -> ML-ready dataset 


## Run the pipelines 

Run slope stability pipeline 

```
python3 subsystems/dag/run_pipeline.py  \
  --pipeline slope_eda \
  --config subsystems/dag/config.yaml

```

```
python3 subsystems/dag/run_pipeline.py \
  --pipeline slope_features \
  --config subsystems/dag/config.yaml
```
# 

```
python3 subsystems/dag/debug_run.py --config config.yaml --pipeline amd
```

Run slope stability pipeline 
```
python3 subsystems/dag/debug_run.py --config config.yaml --pipeline slope 
```


## Testing 

```
python3 -m pytest subsystems/dag/tests/test_amd_pipeline.py 

python3 -m pytest subsystems/dag/tests/test_slope_pipeline.py  

```

```
cd docker
docker compose up --build
docker exec gaiatesting python3 -m pytest /opt/gaia_tsf/subsystems/dag/tests -v
```
