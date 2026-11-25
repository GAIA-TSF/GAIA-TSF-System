# Changelog

Behaviour changes of the ALE engines relative to the moved prototype. Each entry
states the expected difference in outputs.

## Unreleased

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
