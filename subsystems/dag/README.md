# GAIA-TSF – Data Aggregation (DAG) Subsystem

## Overview

The **Data Aggregation (DAG) subsystem** integrates multi-temporal Earth Observation (EO) and auxiliary data into **analysis-ready, machine learning–compatible datasets**.

It serves as a **feature abstraction layer** between raw EO data (Sentinel-1, Sentinel-2) and the MAP (Modeling and Prediction) subsystem.

---

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

---

## Inputs

- **Slope Stability (KV1)**
  - Sentinel-1 LOS displacement time series
  - AOI polygon

- **AMD (KV2)**
  - Sentinel-2 multispectral time series
  - AOI polygon
  - Water mask

---

## Outputs

- ML-ready tensors:
  - `(T, H, W, C)` spatiotemporal feature cubes
- Derived features:
  - velocity, acceleration (slope stability)
  - AMD index and spectral features
- Ready for probabilistic anomaly detection (MAP subsystem)

---

## Architecture

See: `./docs/coding_notes.md` 

**Workflow:** Raw EO Data -> Ingestion -> Harmonization (spatial / temporal) -> Masking (AOI, water) -> Feature Engineering -> Preprocessing -> Validation -> Tensorization -> ML-ready dataset 


## Testing 
```
cd docker
docker compose up --build
docker exec gaiatesting python3 -m pytest /opt/gaia_tsf/subsystems/dag/tests -v
```
