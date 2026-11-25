# Changelog

Behaviour changes of the ALE engines relative to the moved prototype. Each entry
states the expected difference in outputs.

## Unreleased

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
