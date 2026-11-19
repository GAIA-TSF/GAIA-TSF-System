# ALE prototype (two alert engines): Phase 0 plan, revision 2

Status: **draft for review. No code has been changed.**
Branch: `ale-dev`. Date: 2026-11-19.

The spec's `map/ale-3levels/` is `subsystems/map/ale-3levels/` here (package
`ale_3levels`). The three-level prototype to move is in
**`subsystems/map/monitoring/`** (confirmed). It will later move to
`subsystems/ale/`, which today holds only a placeholder README.

**What changed since revision 1:** `map-dev` already has a MAP forecast contract.
It writes a long-format forecast/residual table with `period` and `sample_role`
(§3). The plan now builds on `map-dev` instead of `ale-dev`'s old
`monitoring_input.npz` hand-off, and the decisions in §8 are rewritten.

---

## 1. Branch and code state

| Ref | What it holds |
|---|---|
| `develop` (default) | Old MAP without the forecast contract or the L1/L2/L3 prototype. |
| `map-dev` @ `61f8c7e` | Current MAP: forecasting only. All monitoring code has been removed. `utils/forecasts.py` writes the long-format forecast tables (schema 2.0). Adds `tuning_pipeline.py`, `run_compare.py`, `tests/conftest.py`. |
| `ale-dev` @ `f11c5f2` | Based on `950702d` (the parent of `map-dev`). One WIP commit adding prototype files `monitoring/level1.py`, `level2.py`, `level3.py`, `quality.py`, `series_export.py`, `utils/contract.py`, `utils/forecast_timing.py` and their tests. The old monitoring code is still present. |
| `stash@{0}` | The rest of the prototype: edits to `anomaly_detection.py`, `temporal_monitoring.py`, `monitoring_pipeline.py`, `dashboard.py`, visualisation scripts, plus unrelated MAP/DAG edits. |

Tests: `ale-dev` gives 58 passed, 11 failed, 1 error. `ale-dev` + stash gives 68 passed, 1 error.

**Q1 answer:** the three levels are a prototype, not a validated baseline. B1's
golden test therefore guards the *move* (same inputs give the same numbers before
and after), not correctness.

**Proposed branch operation (needs approval):**
1. `git merge map-dev` into `ale-dev`. This keeps history. The prototype files from
   `f11c5f2` survive as orphans under `subsystems/map/monitoring/`.
2. One commit restores the prototype modules from the stash: only the files under
   `subsystems/map/monitoring/`, their tests and the visualisation scripts, as a
   frozen snapshot. The stash's DAG edits and other MAP edits are **not** taken.
   The stash itself is left as it is.
3. B1 then moves the snapshot into `ale-3levels/engines/three_level/` and
   deletes it from `subsystems/map/`.

---

## 2. Inventory of the prototype (`subsystems/map/monitoring/`)

### 2.1 Stays in MAP (as on `map-dev`)

Datasets and features (`dataset/`); model plugins `const`, `trf`, `lstm` with the
stable-pixel selector; learning, tuning and inference pipelines; `utils/forecasts.py`
(forecast tables, error metrics); explainability; diagnostic plots. MAP already runs
without any alert code.

### 2.2 Moves to ALE (prototype snapshot)

**Z-score formulas.** There are two, and they live in different prototype states:

| | Formula | Where | Fit population |
|---|---|---|---|
| **Z-A (legacy)** | `z = (r − mean(R)) / std(R)`, ddof 0 | `anomaly_detection.py::StatisticalAnomalyDetector.detect` (committed) | every finite residual in the stack, **monitoring included** (look-ahead) |
| **Z-B (WIP)** | `z = r / σ_cal`, no demeaning, ddof 1 | `level1.py::level1_scores`, `frozen_residual_calibration` (stash wires it in) | pooled TSF-mask residuals, pre-monitoring dates only, frozen with a fingerprint |

B2.1's unified `(e − μ_cal)/σ_cal` is a third variant. The prototype also has
internal acceleration z-scores (CUSUM/regime) in `temporal_monitoring.py`. In the
committed version, `_calculate_cusum_signal` takes the std from the non-directional
trend and the mean from the directional trend (l.407–413).

| Concern | File / symbols | Config keys (MAP `config.yaml`, stash) |
|---|---|---|
| L1 | `anomaly_detection.py::StatisticalAnomalyDetector` (`detect`, `write`, `_persistent`); `level1.py` (`level1_scores`, `residual_scale`, `frozen_residual_calibration`) | `anomaly_detection.{zscore_threshold, residual_threshold, direction, persistence, combination_operator, calibration_file}` |
| L2 residual CUSUM | `level2.py` (`Level2Parameters`, `residual_cusum`, `level2_alarms`, `aggregate_residual`, `frozen_level2_calibration`, `compute_level2`, `pixel_level2`, `level2_metadata`, `calibration_replay`, `warn_on_bias`) | `level2.{reference_value, decision_threshold, direction, bias_warning_ratio, calibration_file}` |
| Acceleration CUSUM (supporting evidence) | `temporal_monitoring.py::TemporalResidualMonitor` (`analyze`, `_calculate_cusum_signal`, `_cusum`, `_cusum_spatial_series`, `_regional_cusum`, `spatial_persistent_acceleration`, `spatial_coherent_acceleration_cusum`) | `monitoring.dashboard.cusum.*` / `acceleration_cusum.*` |
| L3 regime score (`P_regime` / `regime_risk`) | `temporal_monitoring.py::_calculate_regime_risk`; `level3.py` (`Level3Parameters`, `sustained`, `compute_level3`, `frozen_level3_calibration`, `level3_metadata`) | `monitoring.dashboard.regime.*`; `level3.{watch_threshold, alert_threshold, consecutive_acquisitions, calibration_file}` |
| Acceleration features for L3 | `temporal_monitoring.py::_causal_gradient[_stack]`, `_ema[_stack]`, `_fill_temporal_gaps`, `_sign_persistence`, `_days_from_start` | `derivative_window`, `smoothing_span` |
| Spatial coherence and persistence | `spatial_coherence.py::SpatialCoherenceDetector`, `SpatialCoherenceRegion` | `monitoring.dashboard.spatial_coherence.{enabled, minimum_area_pixels, persistence, minimum_overlap, connectivity}`, `spatial_support.*` |
| Quality | `quality.py` | `quality.*` |
| Series export | `series_export.py` | — |
| Rasters / orchestration | `anomaly_detection.write`, `residual_analysis.py` writers, `pipelines/monitoring_pipeline.py`, `run_monitoring.py` | `monitoring.*` |
| Plots / animations | `dashboard.py`, `scripts/visualisation/*`, `scripts/simulate_monitoring_animation.py`. Several of these **recompute** results; they will be rewritten to read only saved ALE output. | `monitoring.animation.*`, `plotting.*` |
| Calibration / contract helpers | `utils/contract.py`, `utils/forecast_timing.py` | — |

---

## 3. Storage, dependencies, conventions

- **Storage:** `.npz` plus JSON sidecars everywhere (Q4). GeoTIFF is used for MAP
  diagnostics. CSV is used for tables. No new dependencies; xarray is not used.
- **Python:** system `python3` 3.12. There is no import-linter, so a pytest test
  enforces the import boundary.
- **LOS sign:** negative = motion away from the satellite (confirmed, Q10).
- **Reference:** cumulative LOS = 0 at the first acquisition. Real rasters carry
  `spatial_reference` (`unreferenced` / `stable_area`) and `temporal_reference_utc`
  tags. There is no versioning yet (Q6).
- **Units (Q7):** the synthetic generators (the repo's
  `dag/scripts/synthetic_tsf_deformation.py` and the attached script) write
  **metres** (`/1000`, STAC `simulation:unit: m`). Real sites are in mm. The
  `map-dev` table for the synthetic site declares `target_units: mm` while its
  values are in metres. This is a mislabel that the contract must correct.
- **Cadence:** Sentinel-1, nominally 12 days. Real sites have 6/24/36/60/72-day gaps.
  The synthetic site is strictly 12-day, 122 dates, 2017-01-01 → 2020-12-23.

---

## 4. The MAP forecast table on `map-dev` (`utils/forecasts.py`, schema 2.0)

MAP inference writes `<experiment>/results/.../forecasts/{forecasts,residuals}.npz`
plus `.json`. These cover every valid origin in `mask_file` (the TSF) at leads
1..`horizon`. Learning also writes `models/<exp>/reference_forecasts.npz` for the
stable reference population. Each row is one `(series, forecast_origin, lead)`.

Columns: `series_id, row, col, x, y, forecast_origin_date, lead, target_date,
target_date_status (acquired|expected), last_observation_date, persistence_value,
predicted_value, prediction_lower, prediction_upper, period (calibration|monitoring|outside),
sample_role (train|validation|test|none), observed_value, residual`.
The sidecar has `schema_version, model, model_version (e.g. slope_trf@51fd24bbbf12),
target_variable, target_feature, target_units, horizon, grid{crs, transform, height, width}`.

On the synthetic site (`slope_trf`, temporal split, target `cumulative_los_current`):

| period | sample_role | target dates | n dates |
|---|---|---|---|
| calibration | train | 2017-03-14 → 2018-05-08 | 36 |
| calibration | validation | 2018-06-01 → 2018-08-24 | 8 |
| calibration | test | 2018-09-17 → 2018-12-10 | 8 |

All 14,024 reference pixels are covered. There is no `results/forecasts/` for the
monitoring period yet, so MAP inference (`map-dev`) has to be run once (§7, step 2).

---

## 5. Hand-off contract (ALE side: `contract.py`)

ALE reads the MAP tables (residuals `.npz` plus `.json`). It never imports MAP. The
contract module validates the schema and **harmonizes** the data (Q2):

| Contract field | Source | Notes |
|---|---|---|
| `site_id` | experiment dir name | new sidecar key, written by MAP |
| grid / CRS | sidecar `grid` | |
| `model_id`, `model_version` | sidecar `model`, `model_version` | already present (`trf`, `lstm`; Q8) |
| `tsf_version` | — | new sidecar key. Provisional content hash of the LOS input file list and bytes, because there is no versioning yet (Q6) |
| `forecast_origin`, `target_date`, `lead_step` | `forecast_origin_date`, `target_date`, `lead` | rows with `target_date_status == expected` are excluded from scoring |
| `observed_mm`, `predicted_mm`, `residual_mm` | `observed_value`, `predicted_value`, `residual` | cumulative LOS, converted to mm |
| `observed_mm_per_day`, `predicted_mm_per_day`, `residual_mm_per_day` | `(value − persistence_value) / Δt`, with `Δt = target_date − last_observation_date` in days | rate over the forecast interval; exact, needs no extra input |
| `delta_t_days` | derived | |
| `valid_mask` | finite observed ∧ inside `mask_file` | no coherence raster exists for the synthetic site |
| `period` | `period` | temporal split (Q3) |
| `sample_role` | `sample_role` | spatial or temporal model split (Q3) |
| `split` (spec field) | derived: `monitoring` if `period == monitoring`; `train`/`validation` if that is the `sample_role`; otherwise `calibration` (test rows and pixels never used in fitting) | |
| global attrs | sidecar | `schema_version`, `los_sign_convention = negative_away_from_satellite`, `reference_date`, `reference_point`, `units = mm, mm/day` |

**Units (Q2, Q7, "harmonize"):** ALE always works in **mm** and **mm/day**. The
source unit comes from the LOS input metadata (`units` tag or STAC
`simulation:unit`). If it disagrees with the sidecar's `target_units`, MAP's writer
fixes the sidecar. The ALE reader refuses tables without a trustworthy unit. The
fix in MAP is a small change to the sidecar writer only; models are not retrained.

When the target is velocity rather than cumulative LOS, the cumulative fields come
from a minimal MAP export, `observed_series.npz`: raw cumulative LOS per pixel and
date, in mm, with no derived features. With the current `cumulative_los` target,
`observed_value` and `persistence_value` already give the observed series.

Immutability: MAP writes tables with exclusive creation, named by
`model_version` and `tsf_version`. Reprocessing creates new files.

---

## 6. Target layout

```
subsystems/map/ale-3levels/
  pyproject.toml           # ale_3levels; numpy, pyyaml, rasterio, shapely; matplotlib optional (plots)
  ALE_PLAN.md, README.md, CHANGELOG.md
  config/ale.yaml          # engine, I/O, spatial (SpatialCoherenceDetector params), sectors
  config/fixed_gates.yaml, config/three_level.yaml
  ale_3levels/
    __main__.py  contract.py  outputs.py  spatial.py  state.py  compare.py
    engines/base.py  engines/fixed_gates.py
    engines/three_level/calibration.py  level1_zscore.py  level2_cusum.py  level3_regime.py  quality.py
    plots/                 # read saved outputs only
  tests/  tests/golden/
scripts/run_pipeline.py    # MAP then ALE via their CLIs; imports neither package
```

---

## 7. Commit sequence (after approval)

1. Merge `map-dev` into `ale-dev`; commit the prototype snapshot (§1).
2. **Golden capture.** Run `map-dev` inference on the synthetic site. This writes
   `results/forecasts/` into the ownCloud experiment directory. Then build the
   prototype's input stacks from it (lead 1, mm/day, TSF grid), run the prototype
   L1/L2/L3, the acceleration CUSUM, the regime score and spatial coherence, and
   save the outputs to `tests/golden/`.
   - A cropped/subset extract of the inputs and outputs is committed so the test
     runs without ownCloud.
   - The full-site variant is `@slow` and skips when the data is absent.
3. Scaffold `ale_3levels`: `contract.py` (reader, validation, harmonization),
   import-boundary test. MAP side: sidecar unit/`site_id`/`tsf_version` fix, plus
   the `observed_series` export for velocity targets.
4. **B1:** move the prototype into `engines/three_level/`, with Z-A and Z-B behind
   `zscore_formula: legacy_global | frozen_sigma`. Tolerance: `rtol=1e-12, atol=1e-12`,
   since the same float64 operations run in the same order. Delete the snapshot
   from MAP; MAP-alone test.
5. Engine A plus its tests.
6. `spatial.py`, `outputs.py`, CLI `run`.
7. B2.1 to B2.6, one commit each, each with a CHANGELOG entry.
8. `state.py`, CLI `calibrate`.
9. `compare.py` and the report on the synthetic site; README.

---

## 8. Decisions (2026-11-19)

- **Q1:** the three levels are prototype code. The golden test guards the move only.
- **Q2:** harmonize on `*_mm` (cumulative) and `*_mm_per_day` (rates) in the contract.
  - Engine B (prototype design: velocity residuals) uses `residual_mm_per_day` by default.
  - Engine A uses `observed_mm`.
  - Engine A with `gate_input: residual` uses the 30-day **integrated** residual,
    `Σ residual_mm_per_day · Δt` over the window, with the first interval
    pro-rated at `t − 30 d`. That makes it mm, and the same gates apply.
- **Q3:** two splits.
  - `period` is temporal: calibration = ML training phase, monitoring = inference
    phase where ALE monitors.
  - `sample_role` comes from MAP's `split.method: temporal | spatial` (train,
    validation, test).
  - B2.2 fits μ/σ only on `split == calibration`, i.e. calibration-period rows that
    were neither trained on nor used for early stopping or validation.
- **Q4:** NumPy `.npz` plus JSON sidecars.
- **Q5:** golden and reference data: the synthetic external-trigger site.
  `static/roi_failure_mask.tif` gives ground truth for the comparison report
  (detection lead time).
- **Q6:** no versioning yet. `tsf_version` is a provisional content hash, documented as such.
- **Q7:** synthetic data is in metres. It is converted to mm, and MAP's sidecar unit is corrected.
- **Q8:** `model_id ∈ {trf, lstm}`. Each has its own calibration.
- **Q9 (my choice):** velocity doubling.
  - Gates evaluate whenever the 30-day window is available.
  - Doubling is evaluated only when the prior window (`t − 60 d` → `t − 30 d`) is
    also available within tolerance, **and** `rate(t − 30 d) ≥ velocity_doubling_min_prev_rate_mm`
    (default 1.0, the GREEN boundary). Otherwise a near-zero prior rate would turn
    any jump to 4 mm into RED.
  - When doubling is skipped, the reason goes in `trigger_reason`
    (`doubling_skipped:no_prior_window` / `:prior_below_floor`), and the gate level stands.
- **Q10:** `direction: negative` gates on `max(0, −Δd)`, i.e. motion away from the satellite.
- **Q11:** B native level is `L3` if L3 = alert; `L2` if an L2 alarm or L3 = watch;
  `L1` if an L1 flag; else `NONE`.
  - Severity: WHITE/GREEN/YELLOW/RED and NONE/L1/L2/L3 → 0/1/2/3; `NO_DATA` → null.
  - Per pixel: pixel L1 and pixel L2; TSF/regional L3 is broadcast to the pixels of its support.
- **Q12:** sector aggregation uses `SpatialCoherenceDetector`.
  - For each level L, the binary stack "pixel severity ≥ L" goes through the detector
    (`minimum_area_pixels`, `persistence`, `minimum_overlap`, `connectivity`).
  - The sector level at t is the highest L with a qualified region intersecting the
    sector (whole grid if no polygons).
  - This replaces the spec's `min_pixels`/`min_fraction`/`persistence_acquisitions`.
    Pixel counts per level per sector are still reported.

- **P1 (done):** `map-dev` merged into `ale-dev` (`cd39c03`); prototype snapshot committed (`37f7e2b`).
- **P2:** golden capture and MAP inference run on a **scratch copy** of the synthetic site.
  The ownCloud experiment directory is not written.
- **P3:** engine B calibration on the synthetic site uses a **pooled σ** with a
  `short_calibration` warning. `split == calibration` covers only 10 dates (8 test
  dates + 2 gap dates between the train/validation/test blocks).

## 9. Open points as of revision 2 (now resolved; kept for the record)

**P1 — Branch operation.** OK to merge `map-dev` into `ale-dev` and commit the prototype
snapshot from the stash, limited to `subsystems/map/monitoring/` plus its tests and
visualisation scripts (§1)?

**P2 — Writing into the experiment directory.** Step 2 runs MAP inference, which writes
`results/forecasts/` into the ownCloud site directory, and that syncs. OK? The
alternative is to point `experiment_dir` at a scratch copy of the site.

**P3 — Thin honest calibration on the synthetic site.** With the temporal split,
`split == calibration` on reference pixels is only the 8 **test** dates
(2018-09-17 → 2018-12-10), plus calibration-period rows of any TSF pixels outside
the reference population. That is too few for per-pixel σ, and below the prototype's
`minimum_calibration_samples: 20`. I propose `pooled` calibration as the default
for this site, with a `short_calibration` warning. The alternative is retraining
with a spatial split. Which do you prefer?
