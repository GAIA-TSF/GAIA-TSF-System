# InSAR run diagnostics

Start with summary.json for status, failed stage, configuration and library versions.
Stage timings include work performed in that stage; lazy computations may execute later.
CSV and JSON tables contain the same records. Missing values are JSON null.

- sbas_network.png: date versus perpendicular baseline; edges are selected pairs.
- network_candidates: threshold combinations and their graph component counts.
- acquisition_degree: pair count at each date; single-edge dates lack redundancy.
- slc_coverage: valid aligned input pixels by date.
- wrapped_phase / unwrapped_phase: pair coverage and mean accepted coherence.
- sample PNGs: first and least-covered pair, in radar coordinates; phase is radians.
- reference_phase: combined reference pixel counts and phase offsets by pair.
- displacement_by_date: whole-grid cumulative LOS statistics in mm.

Means can change with valid pixel support. Phase sample colour scales are independent.
These are diagnostics, not uncertainty estimates, component/closure validation,
or complete intermediate phase checkpoints. A partial run has only completed reports.
