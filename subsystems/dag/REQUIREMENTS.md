# DAG requirements traceability

This matrix records the implementation state of the Data Aggregation (DAG)
requirements. “In review” means an implementation and verification artifact
exist but still require formal review. “In development” identifies a partial
implementation or an integration gap.

| ID | Requirement | Verification | Status | Implementation evidence |
|---|---|---|---|---|
| DA_R_01 | Transform raw time series and multi-temporal satellite stacks from the SDI into model-ready features. | Test | In review | Configuration-driven LOS, temporal, meteorological, and topographic pipelines produce aligned GeoTIFF features and JSON metadata. Covered by `test_interfaces.py`, `test_meteo_pipeline.py`, and `test_topographic_features.py`. |
| DA_R_02 | Temporally harmonize remote-sensing observations and in-situ reference samples. | Test | In development | Meteorological features are causally sampled on InSAR dates, and InSAR/in-situ validation pairs exact acquisition dates. The generic `harmonization.temporal` configuration remains disabled and interpolation/nearest matching for in-situ observations is not implemented. |
| DA_R_03 | Spatially overlay in-situ locations and sample surrounding pixels for co-location. | Test | In review | `compare_insar_insitu.py` transforms in-situ WGS84 coordinates to the raster CRS and calculates a configurable nodata-aware neighborhood mean (3×3 by default). Covered by `test_extract_synthetic_insitu.py`. |
| DA_R_04 | Combine spectral data and indices such as NDVI, NDWI, and NDSI with contextual DEM and slope features. | Test | In development | DEM, slope, and PI contextual features are implemented by `topographic_features`. Sentinel-2 ingestion and NDVI/NDWI/NDSI generation are not yet implemented in the DAG subsystem. |
| DA_R_05 | Normalize features using Min-Max or Z-score scaling. | Test | In review | Opt-in `preprocessing.normalization` supports `minmax` and `zscore`; default is disabled. Covered by `test_normalization.py`. |
| DA_R_06 | Use harmonizers to remove unnecessary fields and expose unified structures required by ML models. | Inspection | In development | Raster outputs use a common CRS/grid/profile contract and masks remove invalid spatial samples. A dedicated configurable field-truncation harmonizer is not implemented. |
| DA_R_07 | Bridge data aggregation and model training for training and inference workflows. | Inspection | In development | DAG GeoTIFFs are consumed by MAP `FeatureLoader`. Remaining gaps: MAP learning does not search every configured feature directory, training and inference currently disagree on the model-artifact path, and preprocessing parameters are not fitted exclusively on training data and persisted for inference. |
| DA_R_09 | Handle missing values by imputation or removal. | Test | In review | Opt-in mean/median imputation and consistent incomplete-sample dropping are implemented; default is disabled. Structural mask nodata is preserved. Covered by `test_missing_values.py`. |
| DA_R_10 | Apply logarithmic transformations to skewed data or outliers. | Test | In review | Opt-in signed or unsigned `log1p` and quantile clipping are implemented; default is disabled. Covered by `test_outliers.py`. |
| DA_IR_02 | Provide engineered feature tensors and label vectors to MAP for training and inference. | Test | Not relevant at DAG interface | DAG provides filesystem-level, metadata-described GeoTIFF features. MAP owns conversion to in-memory feature tensors and target vectors through `FeatureLoader` and `DatasetBuilder`. |

## Verification

Run the DAG test suite from the repository root:

```bash
PYTHONPATH=. pytest subsystems/dag/tests
```

The matrix describes implemented behavior rather than test execution in a
particular environment. Formal review should record the test environment,
result, date, and reviewer separately.
