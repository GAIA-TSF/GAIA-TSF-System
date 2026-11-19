# Golden outputs of the three-level prototype

These files pin the behaviour of the alert prototype before it moves into ALE
engine B (plan step B1). They record what the prototype computes, not a validated
result. The prototype is research code, so the golden test only checks that the
move leaves the numbers unchanged.

## `synthetic_crop/`

- **Site:** synthetic external-trigger site (`slope_synthetic_western_platinum_external_trigger`),
  rows 80–111 and cols 64–95 (32×32 px). The crop contains the failure ROI (rows 86–107,
  cols 68–91) plus some stable margin. 116 acquisitions, 2017-03-14 → 2020-12-23.
- **MAP input:** `forecasts/slope_trf/residuals.npz` from `map-dev` inference with model
  `slope_trf@51fd24bbbf12`. Target `cumulative_los_current`, lead 1. The source unit is
  **metres**, converted to mm. Rates are mm/day over the forecast interval.
- **Prototype:** `subsystems/map/monitoring` at `f11c5f2` + `stash@{0}`, run unmodified
  through `run_monitoring.py`, with the dashboard and animation disabled.
- **Legacy Z-A:** `950702d:subsystems/map/monitoring/anomaly_detection.py`, run with the
  `950702d` config (`residual_threshold 0.1, zscore_threshold 1.5, direction negative,
  persistence 3`).
- **Prototype windows:** calibration = all dates before 2018-12-22, monitoring = from 2018-12-22.
  These include training dates. That is the prototype's behaviour and step B2.2 changes it.

| File | Content | Precision |
|---|---|---|
| `inputs.npz` | prototype `monitoring_input.npz` (observed/prediction/residual mm/day stacks, mask, dates, windows) | float64 |
| `rasters.npz` | per-date `anomaly_score`, `anomaly_binary`, `anomaly_l1_binary`, `anomaly_coherent`, `residual_z`; `level2_maximum_instability_cusum`, `level2_first_alarm_index`, `maximum_coherence_acceleration_cusum` | float32 (as the prototype writes GeoTIFFs) |
| `monitoring_series.json` | per-date L1 counts, L2 z/CUSUM/alarm, L3 score/level, quality | float64 |
| `l{1,2,3}_calibration.json`, `monitoring_metadata.json`, `anomaly_summary.json`, `spatial_coherence_summary.json` | calibrations and summaries | float64 |
| `legacy_za.npz`, `legacy_za_summary.json` | Z-A `score`, `binary` | float64 |
| `manifest.json`, `config.yaml` | provenance and the exact prototype config | |

Two capture runs gave byte-identical numbers.

## Reproduce

```bash
git worktree add --detach <proto> f11c5f2 && git -C <proto> stash apply stash@{0}
git show 950702d:subsystems/map/monitoring/anomaly_detection.py > <legacy.py>
python3 subsystems/map/ale-3levels/tests/golden/make_golden.py \
  --residuals <site>/results/forecasts/slope_trf/residuals.npz \
  --mask <site>/static/tsf_mask.tif --source-unit m \
  --prototype <proto> --legacy-anomaly-detection <legacy.py> \
  --crop 80 112 64 96 --run-dir <empty dir> --out <golden dir>
```

Omit `--crop` to capture the full 115×155 grid (about 40 MB, not committed).
