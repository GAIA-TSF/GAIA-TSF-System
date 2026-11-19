# Changelog

Behaviour changes of the ALE engines relative to the moved prototype. Each entry
states the expected difference in outputs.

## Unreleased

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
