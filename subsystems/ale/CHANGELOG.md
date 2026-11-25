# Changelog

Behaviour changes of the ALE engines relative to the moved prototype. Each entry
states the expected difference in outputs.

## Unreleased

- **B2.6 Persisted CUSUM state** (`state.py`, `run --state DIR`). Each
  engine-B run writes `<out>/three_level/state/` (model identity,
  calibration fingerprints, last date, reset history, TSF and per-pixel
  CUSUM accumulators). A later run with `--state` re-applies earlier resets
  and checks that the recomputed accumulators at the previous last date
  equal the stored ones (`StateMismatchError` otherwise; CLI exit 1). CUSUM
  accumulators (TSF, regional, per pixel) restart from zero at the first new
  date after a recalibration or a model-version change, and after each L3
  alert when `state.reset_after_l3: true` (default false = the prototype's
  never-reset behaviour). After a recalibration/model change the output
  recomputes the history with the new calibration/model; issued values stay
  in the earlier output.
  Expected: no change with the default; with reset_after_l3 lower L2
  accumulators after L3 alerts. Observed on the synthetic site: default
  outputs identical to B2.1 (final TSF CUSUM 190.8, no resets);
  reset_after_l3 = true (experiment, config in ALE-DEV/configs): 38 resets
  from 2019-07-02, maximum TSF L2 CUSUM 190.8 -> 17.9, failure-area L3
  unchanged (21.5 %), L2 on the rest of the TSF 87.0 % -> 85.4 %, control
  area false alerts >= L1/L2 52 -> 49 dates.
- **B2.5 Model selection.** `input.residuals` may map MAP models to their
  tables (`{trf: ..., lstm: ...}`); `input.model_id` or `--model` selects the
  model that drives the alerts. Each model has its own engine-B calibration
  (`calibration/<model_id>/lead<k>/`); one model's calibration cannot serve
  another. Expected: no change for a single-model site. Observed on the
  synthetic site (only `trf` exists): both engines' outputs identical to the
  B2.1 run.
- **B2.4 Rename.** Engine B's L3 output is `regime_change_score` (a
  heuristic score in [0, 1), not a probability). `P_regime` and the older
  `regime_risk` are documented legacy aliases (`alerts.json:
  metric_aliases`, `outputs.get_metric`); the prototype's deprecated
  `medium_risk_threshold`/`high_risk_threshold` keys still map to
  `level3.watch_threshold`/`alert_threshold` with a warning. Expected and
  observed: no change to any result (only `alerts.json` gains
  `metric_aliases`).
- **B2.3 Model-version guard.** Frozen engine-B calibrations record their
  model identity (`model_id`, `model_version`, `lead`) and live under
  `calibration/<model_id>/lead<k>/`. `run --calibration DIR` scores with a
  frozen calibration only if it exists (`MissingCalibrationError`) and its
  identity matches the residuals (`CalibrationMismatchError`, naming both
  versions; CLI exit code 1). Expected: no change to results of a normal run.
  Observed on the synthetic site: every output array identical to the B2.1
  run; calibration identity `trf` / `slope_trf@51fd24bbbf12` / lead 1.
- **B2.1 Unified z-score** (`zscore_formula: unified`, new default):
  `z = (e - mu_cal) / sigma_cal` for every engine-B residual z (L1, pixel
  CUSUM, TSF-wide L2, regional L2), each with the frozen calibration mean and
  sigma of its own series (fitted before monitoring; with B2.2 on
  independent records). `frozen_sigma` keeps the prototype's `e / sigma_cal`
  and pins the golden test; `legacy_global` adds the 950702d global z for
  comparison. The L3 regime score and the acceleration CUSUM were already
  centred. Expected: small changes, in the direction of the removed bias.
  Observed on the synthetic site (vs B2.2): L1 mu/sigma = -0.004, L2 bias
  ratio -0.121 (now removed); negative L1 exceedances 81,160 -> 80,620
  pixel-dates (-0.7 %); first L2 (2019-03-28) and first L3 (2019-06-20)
  unchanged; area-level ground-truth results unchanged; pixel shares
  failure area L2 65.5 % -> 65.6 %, L3 21.6 % -> 21.5 %. Engine A unchanged.
- **B2.2 Independent calibration** (`calibration.population: independent`,
  new default; `prototype` keeps the old behaviour and pins the golden test).
  Every engine-B baseline (L1 and L2 sigma, regional L2 sigma, L3 regime
  mean/sigma, acceleration-CUSUM baseline) uses only `split == calibration`
  records: calibration-period records never used for training, validation or
  early stopping. Pooled sigma. `n_calibration` is recorded; fewer than
  `calibration.minimum_dates` (20) logs a warning and raises the
  `short_calibration` quality flag. Limitation: the TSF-level L3/acceleration
  baselines are taken at calibration-split dates, but their causal derivative
  (3 acquisitions) still looks back at up to two earlier dates.
  Expected: larger sigma, fewer and later L1 alerts.
  Observed on the synthetic site (54 -> 10 calibration dates; 757,296 ->
  140,240 L1 records): L1 sigma 0.3565 -> 0.3576 mm/day (+0.3 %), negative
  |z| > 1.5 pixel-dates 81,692 -> 81,160 (-0.7 %), first L1 unchanged
  (2018-12-22) — the model's in-sample and held-out residuals have almost the
  same scale, so L1 barely moves. L2 sigma 0.01166 -> 0.01297 (+11 %), first
  L2 2019-03-16 -> 2019-03-28 (12 d later) — as expected. L3 regime sigma
  3.83e-4 -> 2.44e-4 (-36 %; mean -6.7e-5 -> -4.8e-5): *not* as expected; the
  10 independent dates lie mostly in one season (2018-09-17..2018-12-10) and
  underestimate the variability of the model-relative acceleration across
  seasons, so L3 becomes more sensitive: failure-area L3 (area level)
  2019-12-05 -> 2019-11-23 (12 d earlier), control-area L3 false alerts
  0 -> 0; pixel shares failure area L3 20.5 % -> 21.6 %, L2 68.2 % -> 65.5 %.
  Engine A unchanged.
- Engine B: L3 attributed to the coherence-qualified regions active at the
  alert date (region_id k), no longer to the whole TSF (decision
  2026-11-25). Engine-B numbers unchanged; only per-pixel levels move.
  Expected: L3 disappears outside the coherent regions (those pixels fall to
  L2 via the TSF-wide L2 alarm); L3 stays on the failure area where a region
  forms. Observed on the synthetic site (monitoring pixel-dates): failure
  area L3 59.7 % -> 20.5 % (L2 29.0 % -> 68.2 %); rest of the TSF L3
  59.6 % -> 0.0 % (L2 29.0 % -> 88.6 %); all L3 now in region 1 (132 px,
  from 2019-06-08). Area level: failure area L3 unchanged (first 2019-12-05,
  33 dates); control area L3 false alerts 33 -> 0 dates. Engine A unchanged.
- Common output schema and `run` command; sector aggregation. Pixel results
  of both engines unchanged (format only). Engine B no longer writes the
  prototype files (series, pixel_stacks.npz, metadata) during `run`; the
  golden test still produces and compares them. New sector level: highest
  level held by coherence-qualified pixels (SpatialCoherenceDetector per
  level; min_pixels 25 or min_fraction 0.10). Expected: sector levels lag
  pixel levels by the coherence persistence (3 acquisitions). Observed on the
  synthetic site (one sector = whole TSF, monitoring): A GREEN for 3 dates,
  then RED from 2019-01-27 (59 dates); B NONE until 2019-04-09, L2 from
  2019-04-09 (first pixel L2 2019-03-16), L3 from 2019-12-05 (first pixel L3
  2019-06-20; L3 alerts are not persistent for 3 consecutive acquisitions
  until December).
- Engine B in the common schema (`ThreeLevelEngine`): no engine-B number
  changes (golden test unchanged; adapter tests check that z, the L2 CUSUM
  and the L3 score pass through bit-identically). New per-pixel labels
  NONE/L1/L2/L3 with `level_scope` (pixel/region) and `region_id`
  (0 = whole TSF, k = coherent region). Expected: L2/L3 are TSF-level
  decisions in the prototype, so they cover the failure area and the stable
  rest alike; only L1 is localised. Observed on the synthetic site
  (monitoring, all TSF pixels): failure area 59.7 % L3 / 29.0 % L2 / 4.1 % L1
  / 7.2 % NONE; rest of the TSF 59.6 % L3 / 29.0 % L2 / 0.7 % L1 / 10.5 %
  NONE. First L2 2019-03-16, first L3 2019-06-20; one coherent region
  (132 px) from 2019-06-08.
- Engine A `fixed_gates` (new engine). 30-day LOS movement with linear
  interpolation at `t - 30 d` (tolerance 24 d), gates WHITE < 1 <= GREEN < 4 <=
  YELLOW <= 10 < RED, velocity doubling (ratio 2, rate >= 4 mm, previous rate
  >= 1 mm). Expected: RED dominates the failure area; noise (atmospheric sigma
  12 mm on the synthetic site) puts many stable pixels in GREEN/YELLOW.
  Observed on the synthetic site, monitoring period, all TSF pixels, before
  sector aggregation: failure area 94.5 % RED / 3.8 % YELLOW / 1.3 % GREEN /
  0.4 % WHITE; rest of the TSF 10.4 % RED / 18.8 % YELLOW / 49.8 % GREEN /
  20.9 % WHITE / 0.1 % NO_DATA; 81 % of RED pixel-dates come from velocity
  doubling.
- B1 (no behaviour change): the three-level prototype moved from MAP into
  `engines/three_level`. Bit-identical to the prototype on the golden synthetic
  crop (`tests/test_ale_three_level_golden.py`, tolerance exact). The legacy
  global z-score (950702d) is reachable with `zscore_formula: legacy_global`.
  Outputs are `.npz`/JSON/CSV instead of GeoTIFF; dashboard and animations are
  not produced until the plotting step.
- Package moved from `subsystems/map/ale-3levels` to `subsystems/ale`.
- Package scaffold: contract reader (MAP forecast table schema 2.0 → mm and mm/day),
  engine interface and output schema, configuration files, `validate` CLI.
  No engine behaviour yet.
