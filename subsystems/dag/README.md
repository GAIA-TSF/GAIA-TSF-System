# Data Aggregation (DAG) Sub-system 

The **Data Aggregation** sub-system serves as the critical
processing bridge that transforms data stored within the
Spatial Data Infrastructure (SDI) into structured inputs suitable for
machine learning analysis. Its primary function is to prepare data in
structures that are harmonised and ready for downstream consumption,
adhering to the principle that "garbage in, garbage out" dictates
model performance. The implemented workflows ingest Sentinel-1 LOS stacks,
meteorological observations, static terrain, and synthetic in-situ locations.
Sentinel-2 AMD processing calculates a configured two-band indicator before
regional and point-based exploratory analysis.

![Data Agregation Architecture](../../images/dag_subsystem.png)

## Key Capabilities

- Multi-temporal Sentinel-1 LOS ingestion
- Exploratory Data Analysis (EDA): step important to select the following steps 
- Spatial harmonization (resampling to common grid)
- Feature engineering:
  - Slope stability: displacement → velocity → acceleration
  - Static terrain: DEM, slope, and topographic position index (PI)
  - Meteorology: precipitation accumulations/extremes and temperature metrics
- TSF-mask application with grid consistency checks
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
  - Paired multiband Sentinel-2 JP2 images and JSON metadata
  - AMD and clean-water masks, plus their water observation points
  - Configurable band ratio or difference, followed by EDA

## Outputs

- graphs and maps 
- Metadata-described GeoTIFF feature rasters consumed by MAP
- Derived features:
  - velocity, acceleration (slope stability), etc. 
  - Static DEM, slope, and PI contextual features
- Ready for probabilistic anomaly detection (MAP subsystem)

## Architecture

**Implemented workflow:** filesystem inputs → ingestion → grid/date validation →
TSF masking → feature engineering → optional preprocessing → GeoTIFF features
and JSON metadata → MAP feature loading.

## Requirements traceability

The maintained implementation status, verification method, evidence, and known
gaps for DA_R_01–DA_R_10 and DA_IR_02 are recorded in
[REQUIREMENTS.md](REQUIREMENTS.md). The matrix intentionally distinguishes
implemented behavior from planned or partially integrated behavior.

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

### In-situ co-location

Create the in-situ CSV by spatially overlaying every labelled point from the
GeoPackage configured under `in_situ.static.observation_points` on each
configured TRUE_LOS acquisition:

```bash
python3 subsystems/dag/scripts/extract_synthetic_tsf_in-situ_deformations.py \
  --config subsystems/dag/config.yaml
```

The DAG `in_situ.sampling.window_size` setting controls the square, nodata-aware
pixel neighbourhood used for each sample. The script accepts any number of
uniquely labelled observation points. All paths are resolved below the DAG
`project_dir`; use `--project-dir` to override that root.

Run the independent validation step after extraction:

```bash
python3 subsystems/dag/scripts/compare_insar_insitu.py \
  --config subsystems/dag/config.yaml
```

This script samples the configured satellite LOS rasters at the locations and
dates in the in-situ CSV. It writes a two-column
`insar_los,insitu_deformation` comparison CSV and a JSON report containing
sample counts, dataset means, bias, MAE, RMSE, Pearson correlation, and R².
Their paths and the raster-to-output unit scale are configured under
`in_situ.validation`. `sampling_window_size: 3` selects a 3×3 nodata-aware
InSAR mean centred on every in-situ location.


## Run the pipelines 

### AMD index and EDA

Configure the `amd` section in `config.yaml` with the actual `first_band` and
`second_band`.
`method: ratio` calculates first / second; `method: difference` calculates
first − second. This spectral indicator is not a calibrated chemistry measurement.

```bash
python3 subsystems/dag/run_pipeline.py \
  --pipeline amd_index \
  --config subsystems/dag/config.yaml
```

The index stage writes `amd_index.tif` to `amd.results.index.output_dir`
(`results/eda/source_index` in the supplied configuration).
EDA then consumes that generated product:

```bash
python3 subsystems/dag/run_pipeline.py \
  --pipeline amd_eda \
  --config subsystems/dag/config.yaml
```

The pipeline always engineers the intermediate `amd_index.tif` and dated
`metadata.json` before EDA. It writes these outputs to `results/eda`:

- `inventory.csv`: image/JSON pairing, dates, dimensions, CRS, resolution, band
  counts, and inventory status.
- `quality_report.json`: validation errors, unpaired metadata, mask overlap,
  acquisition gaps, scene cloud percentages, and valid index coverage per region.
- `statistics.json`: overall and per-acquisition index statistics for AMD and
  clean-water masks. Empty observations have null statistics and zero counts.
- `point_timeseries.png` and `point_timeseries.csv`: index values at every point
  in `amd_water_point.gpkg` and `clean_water_point.gpkg`. The general
  `observation_points.gpkg` is not required for this two-region analysis.

All configured paths resolve relative to `project_dir`. The AMD mask defines the
output grid; the clean-water mask must match. Both regions are retained. Larger
images on the same pixel grid are cropped without interpolation. Different
CRS/resolution or unaligned origins are rejected. Points are reprojected and must
lie inside their associated mask. `points.window_size` selects an odd pixel
neighbourhood restricted to that mask. Plot lines connect valid observations
across missing acquisitions within each calendar year, with breaks between
years. Missing values remain missing in the CSV and statistics.

`band_positions` contains one-based positions for this repository's DPR export,
including SCL at position 13. Verify these for other exporters. Alternatively,
omit positions when raster band descriptions identify bands unambiguously.
JSON `eo:bands` order is not treated as raster order. Sensing dates are checked
against JSON dates where available; duplicate dates require upstream selection.

Quality filtering excludes configured SCL classes and source nodata. Setting
`quality.scl_band: null` disables SCL filtering and records a warning. Scene cloud
percentages do not replace pixel masks. Zero or near-zero denominators, controlled
by `denominator_epsilon`, produce nodata. No smoothing or gap filling is applied.

Both bands are converted with `stored_value * scale + offset` before calculation.
Defaults preserve stored values; configure conversion for your input product.
DPR already applies band offsets, so do not apply them twice. Ratios are
dimensionless; differences retain the units of the scaled inputs. Feature
metadata records the formula and sources. The output stem is `amd_index`; a
later MAP dataset must select that stem. Lag generation and monitoring remain
separate work.

### AMD model features

Run these stages in order with the same configuration:

```bash
python3 subsystems/dag/run_pipeline.py --pipeline amd_features --config subsystems/dag/config.yaml
python3 subsystems/dag/run_pipeline.py --pipeline amd_temporal_features --config subsystems/dag/config.yaml
python3 subsystems/dag/run_pipeline.py --pipeline amd_meteo_features --config subsystems/dag/config.yaml
```

`amd_features` writes `amd_index.tif` for ratio or `amd_difference.tif` for
difference. It excludes fully invalid dates and dates below
`amd.acquisitions.min_valid_fraction`, evaluated over the union of both water
masks after SCL, source nodata, and denominator filtering. Partially cloudy
pixels remain nodata. Intermediate unfiltered products live in `source_index/`.
The optional `amd.results.index` directory keeps standalone index/EDA products
separate from model features; legacy configurations fall back to `results.features`.

Temporal outputs are `lag1`, `lag2`, `lag3`, `roll_mean`, `roll_std`,
`amd_diff1`, `amd_diff2`, `amd_diff3`, `annual_sin`, and `annual_cos`.
Lags count prior valid observations separately at each pixel; differences are
current value minus the corresponding lag, rather than repeated derivatives.
Rolling windows include the current valid observation, use population standard
deviation, and require the configured `min_periods` (default: full window).
History continues across years. All outputs are nodata at cloudy current pixels.
Seasonal phase uses day of year and a 365.2425-day annual period.

AMD meteorology reads `amd.meteorology.inputs.table`, a consecutive daily CSV,
and reuses the slope meteorological feature extractor. Calendar-day weather
windows include cloudy days; results are sampled on accepted AMD acquisition
dates and masked by the base feature's pixel validity. Blank weather values
remain missing input values under the existing extractor's rules. Enable
`temperature_anomaly` only with an explicit `temperature_baseline` in
`amd.meteorology.feature_engineering`, configured or fitted using training data.
It is disabled by default. The weather table must cover every accepted AMD date
and should include the preceding 60 days for complete longest-window history.

Each stage writes dated multiband GeoTIFFs and `metadata.json` in its configured
output directory: `results/features`, `results/temporal_features`, and
`results/meteo_features` in the supplied configuration. Temporal and weather stages consume the existing base product
and reject changed formula, quality, or acquisition-selection settings. Rerun
`amd_features` after changing its inputs or masks. Select the exact generated
stems and all three output directories when configuring MAP feature loading.
These stages preserve missing observations and do not fit normalization or
imputation parameters; fitted preprocessing belongs to the training workflow.

### Slope stability and contextual features

Feature normalization is disabled by default. Enable it globally for generated
DAG features with either Min-Max scaling or Z-score standardization:

```yaml
preprocessing:
  normalization:
    enabled: true
    method: zscore  # zscore | minmax
    per_feature: true
```

Non-finite pixels remain missing and constant-valued features normalize to
zero. With `per_feature: true`, each feature is scaled independently to prevent
large-range variables from dominating downstream models.

Missing-value handling is also disabled by default. Enable mean or median
imputation, or consistently drop incomplete sample positions across features:

```yaml
preprocessing:
  missing_values:
    enabled: true
    strategy: median  # mean | median | drop
    max_nan_ratio: 0.2
```

Structural nodata outside configured TSF masks remains missing and is excluded
from the missing-ratio calculation.

Logarithmic outlier transformation is likewise opt-in. By default the signed
`log1p` form is used, preserving the direction of negative deformation values:

```yaml
preprocessing:
  outliers:
    enabled: true
    method: log
    features: [precipitation, precip_30d]
    signed_log: true
```

An empty `features` list transforms every generated feature. The same stage
also supports quantile clipping through `method: clip` and `clip_range`.

Generate the static DEM-derived topographic feature set (DEM, slope in degrees,
and PI/topographic position index):

```bash
python3 subsystems/dag/run_pipeline.py \
  --pipeline topographic_features \
  --config subsystems/dag/config.yaml
```

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
