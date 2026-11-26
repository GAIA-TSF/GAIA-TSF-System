# Alert Decision Support Engine (ALE) Sub-system

The **Alert and Decision Support Engine** continuously processes
information from the SDI sub-system, which provides outputs from
machine learning models such as time series trend analyses,
deformation rates, vegetation and land cover changes, and predictions
of water and soil chemical properties. The ALE evaluates these data to
determine whether conditions at a TSF are normal or indicative of an
anomaly or risk. When a threshold is exceeded or an unusual pattern is
detected, the engine categorizes the severity and sends alerts to the
notification system for dissemination to relevant stakeholders.

![Alert Decision Support Engine Architecture](../../images/ale_subsystem.png)

## Prototype: `ale_3levels`

MAP forecasts cumulative LOS displacement and saves its forecasts and residuals;
it makes no alert decision. ALE turns those saved MAP outputs into alerts with
two interchangeable engines that read the same input and write the same output
schema, so they can be compared on the same data:

- **Engine A, `fixed_gates`:** fixed thresholds on the 30-day LOS movement
  (WHITE / GREEN / YELLOW / RED, with velocity doubling).
- **Engine B, `three_level`:** z-score (L1), CUSUM (L2) and regime-change score
  (L3). It was moved out of MAP unchanged (golden test) and then corrected
  (B2: independent calibration, unified z-score, version guard, model
  selection, persisted state).

**Safety.** The engines only emit levels and advisory text. They never execute an
action, call a network service or send a message to any system. Recommended
actions are text from the configuration. Tests check that scoring has no side
effects.

### Layout and boundary

```text
subsystems/ale/
├── ale_3levels/
│   ├── contract.py          MAP -> ALE hand-off: read, validate, harmonize
│   ├── engines/base.py      Engine interface, AlertResult, severity mapping
│   ├── engines/fixed_gates.py
│   ├── engines/three_level/ engine B (moved prototype + B2), engine.py adapter
│   ├── spatial.py           SpatialCoherenceDetector, sector aggregation
│   ├── state.py             persisted engine-B state (B2.6)
│   ├── outputs.py           common output schema and writers
│   ├── run.py               run / calibrate
│   ├── compare.py           comparison report (only reader of ground truth)
│   ├── plots.py             plots and animations from saved results only
│   └── demo.py              small simulated site
├── config/                  ale.yaml, fixed_gates.yaml, three_level.yaml, ale_synthetic.yaml
├── scripts/                 prepare_synthetic.sh, make_demo_data.py
└── tests/                   tests and golden files
```

`ale_3levels` reads saved MAP files only. It never imports MAP, and MAP never
imports it; `tests/test_ale_import_boundary.py` enforces this.
`scripts/run_pipeline.py` (repository root) runs MAP and then ALE as separate
processes. The root `pytest.ini` puts `subsystems/ale` on the import path;
`pip install -e subsystems/ale` also works.

### Hand-off contract (`contract.py`)

Input: MAP's forecast-residual table `results/forecasts/<experiment>/residuals.npz`
with its `residuals.json` sidecar (table schema 2.0). There is one row per
(pixel, forecast origin, lead). The reader validates the schema, fails with a
clear `ContractError` when it does not match, selects one lead and returns dense
`(time, row, col)` stacks:

| Field | Meaning |
|---|---|
| `observed_mm`, `predicted_mm`, `persistence_mm`, `residual_mm` | cumulative LOS in mm; `residual = observed − predicted` |
| `observed_mm_per_day`, `predicted_mm_per_day`, `residual_mm_per_day` | rate over the forecast interval: `(value − persistence) / Δt` |
| `delta_t_days` | `target_date − last_observation_date` |
| `split` | `train`, `validation`, `calibration`, `monitoring`, `outside` or `no_record` |
| `valid_mask`, `mask` | finite record inside the TSF mask |
| `series_dates`, `observed_series_mm` | observed cumulative LOS on every acquisition (engine A) |
| `attrs` | `site_id`, `model_id`, `model_version`, `tsf_version`, units, LOS sign convention, reference date/point, windows, table SHA-256 |

- **`split`** combines MAP's temporal `period` (calibration / monitoring) with
  the model's `sample_role` (train / validation / test). Calibration-period
  records that were neither trained on nor used for validation or early
  stopping are `calibration`.
- **Units.** Everything is converted to mm and mm/day. Synthetic sites are
  written in metres while MAP declares mm: set `input.source_units_override: m`.
- **Sign convention.** Negative LOS means motion away from the satellite.
- **Immutability.** ALE never writes MAP outputs. A reprocessed TSF gets new MAP
  tables; `tsf_version` is provisional (`unversioned-<table hash>`) because no
  processing version exists yet.

### Engine A: `fixed_gates` (`config/fixed_gates.yaml`)

For each pixel and acquisition `t`: `disp_30d_mm = d(t) − d(t − 30 d)`, with
`d(t − 30 d)` linearly interpolated between the two acquisitions on either side.
`rate_mm_per_month` is the same movement scaled to 30 days. `NO_DATA` when no
acquisition lies within `interpolation_tolerance_days` (24) before `t − 30 d`,
when a bracketing value is missing, or outside the TSF mask.

- `gate_input: observed` (default) gates the observed LOS movement; `residual`
  gates the 30-day sum of MAP's one-step residuals (movement the model did not
  expect).
- `direction: abs` (default), `negative` (motion away from the satellite) or
  `positive`.

**Gate boundaries** (on the directional magnitude `m` of `rate_mm_per_month`).
The source specification overlaps at 4 and 10 mm; these boundaries are used:

| Gate | Condition | Recommended actions (text only) |
|---|---|---|
| WHITE | `m < 1` | archive, mark sector stable, wait for the next pass |
| GREEN | `1 ≤ m < 4` | desktop check against rainfall / water table; filter atmospheric and decorrelation noise; no field deployment |
| YELLOW | `4 ≤ m ≤ 10` | site visit within 48 h; cross-check GNSS / inclinometers / total stations; prepare mitigation |
| RED | `m > 10` or velocity doubling | halt heavy operations; evacuate the local zone; prioritise the next 6-day Sentinel-1 pass |

**Velocity doubling:** RED when `m(t) ≥ 2 × m(t − 30 d)`, applied only when
`m(t) ≥ 4 mm` and the previous window's rate `m(t − 30 d) ≥ 1 mm` (all
configurable). The floor on the previous rate stops a near-zero previous rate
from turning any jump into RED. When the previous window is unavailable,
doubling is skipped and the gate level stands; `trigger_reason` records why.

### Engine B: `three_level` (`config/three_level.yaml`)

- **L1** (pixel): `z > zscore_threshold` in the instability direction
  (instantaneous), with a persistence-qualified variant.
- **L2** (TSF and coherent regions): CUSUM of z, alarm when `C > h`.
- **L3** (TSF decision): regime-change score from model-relative acceleration;
  `alert` needs an L2 alarm, a sustained score and a coherence-qualified region.

**z-score.** `zscore_formula: unified` (default) is `z = (e − μ_cal) / σ_cal` for
every residual z, with μ and σ frozen before monitoring. `frozen_sigma` is the
prototype's `e / σ_cal`; `legacy_global` adds the 950702d global z
`(e − mean(all e)) / std(all e)` for comparison only.

**Calibration.**
- `calibration.population: independent` (default): every baseline (L1 and L2 σ,
  regional L2 σ, L3 regime baseline, acceleration CUSUM baseline) uses only
  `split == calibration` records.
- `prototype`: all pre-monitoring records, including model-training dates.
- σ is pooled over the TSF.
- `n_calibration` is recorded. Fewer than `minimum_dates` (20) dates warns and
  raises `short_calibration`.
- Frozen calibrations record `(model_id, model_version, lead)` and live under
  `calibration/<model_id>/lead<k>/`. `run --calibration` refuses another model,
  version or lead (`CalibrationMismatchError`).
- `input.model_id` / `--model` selects the MAP model; each model has its own
  calibration.

**Persisted state** (`state.py`).
- Each run writes `<out>/three_level/state/`.
- `run --state DIR` continues from it. Earlier resets are re-applied, and the
  recomputed accumulators must match the stored ones (`StateMismatchError`).
- CUSUM accumulators restart from zero after a recalibration, after a model
  version change, and after each L3 alert when `state.reset_after_l3: true`
  (default `false`, the prototype's behaviour).

**Pixel levels and `level_scope`.** Engine B decides L2 and L3 for the TSF or for
coherent regions, not per pixel. Each pixel reports where its level came from:

| Level | Source | `level_scope` | `region_id` |
|---|---|---|---|
| L1 | the pixel's own z | pixel | −1 |
| L2 | TSF-wide L2 alarm or L3 watch score | region | 0 (whole TSF) |
| L2 | regional L2 alarm, pixel in an active coherent region | region | k |
| L3 | L3 alert, pixel in an active coherent region | region | k |

**Names.** The L3 output is `regime_change_score`, a heuristic score in [0, 1),
not a probability. `P_regime` and `regime_risk` are documented legacy aliases
(`alerts.json: metric_aliases`, `outputs.get_metric`).

### Common output schema (`outputs.py`)

`<out>/<engine>/` contains:
- `alerts.npz`: per pixel and acquisition: `level`, `severity`,
  `trigger_reason`, `level_scope`, `region_id`, `metric_*`.
- `alerts.json`: `engine_version`, `config_hash`, vocabularies, recommended
  actions and the MAP input identity.
- `history.csv`: per-acquisition counts and metric summaries.
- `sectors.csv` / `sectors.json`.

`<out>/run.json` describes the run. Outputs are written once and
deterministically: reruns are byte-identical.

**Severity, for comparison only.** The levels of the two engines do not mean
the same thing:

| Severity | A `fixed_gates` | B `three_level` |
|---|---|---|
| 0 | WHITE | NONE |
| 1 | GREEN | L1 |
| 2 | YELLOW | L2 |
| 3 | RED | L3 |
| −1 | NO_DATA | NO_DATA |

**Sectors** (`config/ale.yaml: spatial`). For each level L, the pixels at
severity ≥ L pass through `SpatialCoherenceDetector`: connected areas of at
least `minimum_area_pixels` that persist for `persistence` acquisitions with
overlap ≥ `minimum_overlap`. A sector reaches L when its coherent pixels number
at least `min_pixels` or at least `min_fraction` of its valid pixels. Sectors
come from polygons (`sectors_file`) or the whole TSF.

### CLI

Run from `subsystems/ale`, or anywhere with `subsystems/ale` on `PYTHONPATH`:

```bash
python -m ale_3levels validate  --config CONFIG [--input TABLE] [--model trf|lstm]
python -m ale_3levels run       --engine A|B|both --config CONFIG --out DIR \
                                [--input TABLE] [--model M] [--calibration DIR/calibration] [--state PREV/three_level/state]
python -m ale_3levels calibrate --engine three_level --config CONFIG --out DIR [--model M]
python -m ale_3levels compare   --a RUN/fixed_gates --b RUN/three_level --out DIR [--truth roi_failure_mask.tif]
python -m ale_3levels plot      --run RUN --out DIR [--animation] [--no-maps]
python3 scripts/run_pipeline.py --map-config MAP.yaml --ale-config ALE.yaml --out DIR [--truth MASK]   # repository root
```

`--engine A` is `fixed_gates` and `--engine B` is `three_level`; the full names
also work.

Changing any ALE threshold only needs an ALE re-run; MAP outputs are never
touched. The comparison report is the only reader of a ground-truth mask.

### Data (kept outside the repository)

Generated data lives in the ALE data directory, by default
`~/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV` (override with
`ALE_DATA_DIR`). Nothing generated is committed.

For a demonstration of both engines, use the showcase site and follow
[DEMO.md](DEMO.md):

```bash
python3 subsystems/ale/scripts/run_demo.py [--engine A|B|both]   # -> ALE-DEV/showcase
```

```bash
# Small simulated demo site (about 2.5 MB, ~1 s; no MAP, no site data):
python3 subsystems/ale/scripts/make_demo_data.py --run          # -> ALE-DEV/demo
# Real synthetic site: copy from tsf_experiments + MAP inference (~1.5 min):
subsystems/ale/scripts/prepare_synthetic.sh                       # -> ALE-DEV/synthetic_site
cd subsystems/ale
python3 -m ale_3levels run --engine both --config config/ale_synthetic.yaml --out ~/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV/runs/<name>
```

### Tests

```bash
python3 -m pytest subsystems/ale/tests
ALE_SYNTHETIC_TABLE=<residuals.npz> ALE_SYNTHETIC_MASK=<tsf_mask.tif> \
  python3 -m pytest subsystems/ale/tests -m slow   # contract vs golden inputs
```

`tests/golden/` holds the prototype's golden outputs; see its README. The golden
test is pinned to the pre-B2 configuration (`frozen_sigma`, `prototype`
calibration).

### Known limitations

- **Short independent calibration (engine B).** With `calibration.population:
  independent`, engine-B baselines use only calibration-period records the model
  never saw. On the synthetic site (temporal split) that is 10 acquisitions,
  mostly from one season (2018-09-17 to 2018-12-10). They underestimate the
  seasonal variability of the L3 regime signal (σ −36 % against the 54-date
  prototype baseline), so L3 is more sensitive than a full-year baseline would
  make it. The run warns and raises `short_calibration` below
  `calibration.minimum_dates` (20). Accepted for the prototype (2026-11-25); a
  spatial MAP split or a longer independent calibration period removes it.
- **TSF-level L3 and acceleration baselines** are taken at calibration-split
  dates, but their causal derivative (3 acquisitions) still looks back at up to
  two earlier dates, which may be model-training dates.
- **Inherited region levels.** Engine B's L2 is a TSF-wide alarm by design
  (decided 2026-11-25): it signals instability of the facility as a whole, and
  localisation is L3's job. Every TSF pixel inherits it (`level_scope = region`),
  so on the synthetic site the stable control area shows the same L2 dates as the
  failure area; read engine B's L2 false-alert count in the comparison report as
  a facility-level alarm, not a pixel error. L3 is localised to
  coherent regions. The regional L2 series is one combined series (any region
  alarming), so all active regions inherit it.
- **Noise near the 1 mm gate (engine A).** With about 12 mm atmospheric noise
  on the synthetic site, about 21 % of consecutive pixel transitions flip
  between WHITE and GREEN, and the stable control area is GREEN almost
  throughout. YELLOW and RED stay clean at sector level thanks to the coherence
  rule; per pixel, about 9 % of control pixel-dates are RED.
- **`tsf_version` is provisional.** It is a hash of the MAP table, so it changes
  when MAP is rerun (MAP's random forest is not bit-reproducible, about 1e-14 mm/day).
  It should identify the LOS input data instead.
- **Calibration mode.** Only the pooled σ is implemented (`per_pixel` is rejected).
- **Engine B outputs before monitoring** are `NO_DATA`. Engine A also gates
  calibration-period dates; reports use monitoring dates only.
