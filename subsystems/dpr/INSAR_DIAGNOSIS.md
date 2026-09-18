# Cadia and Jagersfontein InSAR investigation — 17 September 2026

The existing displacement maps cannot establish absence of pre-failure movement.
The code contains signal-removal and masking defects. These are corrected, but
the two full SLC stacks have **not** yet been reprocessed or validated against
stable reference ground or field observations.

## Confirmed findings

| Finding | Consequence | Correction |
| --- | --- | --- |
| Unwrapped phase minus a Gaussian surface with wavelength `3 * 30 = 90 m` | Suppresses spatially broad deformation as well as atmospheric contributions; retains short-scale residuals | Preserve phase; optionally subtract only a stable-ground offset per pair |
| `intf.where(corr.where(corr >= 0.20))` | NaN is truthy as a condition, so low-coherence phase survives the intended mask | Explicit finite Boolean mask, also applied after unwrapping and to inversion weights |
| Different intensity and complex-phase smoothing kernels (20 m versus 30 m) | Coherence numerator and denominator do not represent the same averaging operation | Use matching 30 m kernels |
| Arithmetic averaging of wrapped angles | Can average values near +π and −π to zero | Average unit complex phasors |
| `velocity(sol)` where `sol` is radians | Export is radians/year, not mm/year, including a different sign from LOS conversion | Fit velocity after phase-to-mm conversion |
| First displacement epoch replaced with NaNs | Removes the valid temporal origin and compromises residual checks | Preserve zero at the first epoch where connected observations exist |
| Minimum-norm inversion after per-pixel pair masking | Disconnected dates can receive unsupported values | Mark dates outside the first epoch's temporal component as nodata |
| Phase-wrapping RMSE function called on millimetres | Large errors can wrap into small errors | Compute coherence-weighted residual RMSE directly in mm |
| Exports lack explicit units/nodata and quality maps | Makes unreliable pixels harder to identify | Add units, NaN nodata, coherence, valid-pair fraction and RMSE rasters |

The installed processing container uses PyGMTSAR `2025.4.8.post1`. Its source
was inspected directly, including `velocity`, `lstsq1d`, `snaphu`, and `rmse`.
Upstream implementation: [PyGMTSAR](https://github.com/AlexeyPechnikov/pygmtsar).

## Comparison with the supplied notebook

`Jagersfontein_TSF.ipynb` uses the `insardev` stack API, rather than the installed
PyGMTSAR API. Cell 67 fits a trend using
`stack.trend2d(mphase, mcorr, stack.transform()[['azi','rng','ele']])`;
cell 69 subtracts that fitted trend. It does **not** implement DPR's 90 m
Gaussian high-pass. It also includes separate 2D and 1D unwrapping paths.
Its displayed search settings cover July–October 2022, whereas this DPR
project covers 2020–2022. The two are not equivalent experiments.

The correction deliberately does not reproduce an unconstrained spatial trend
fit over the TSF: such a fit could also absorb real deformation. Subtracting a
verified stable-ground offset removes a pair's common offset, but does not
correct spatially varying atmosphere, DEM error, or spatial unwrapping errors.

## Existing products inspected

Both work directories have already been deleted by the original pipeline.
The raw input directories and final rasters remain. Old TIFFs cannot recover
the removed spatial signal; reprocessing is necessary.

| Product | Existing grid | Existing 1st/50th/99th percentiles |
| --- | --- | --- |
| Cadia `disp_20180225.tif` | 434 × 295 | −22.101 / 0.121 / 22.065 mm |
| Jagersfontein `disp_20221228.tif` | 81 × 149 | −35.727 / 0.273 / 34.316 mm |

These are whole-raster statistics, not measurements of the failure footprint.
They show nonzero residuals, not validated deformation. The Jagersfontein
sample above is after the failure, used only to inspect file characteristics.

Cadia has 103 exported dates; Jagersfontein has 90. Cadia's local February
products are 1, 13 and 25 February 2018, followed by 9 March. There is no
9 February product in this output directory. Whether the 9 March acquisition
was before or after failure requires comparing acquisition time with event time.

## Stable reference and rerun

Provide polygon features with a declared CRS per site. Select independently stable
ground outside the tailings, embankments and active excavation/construction.
It must be inside the processing AOI and retain coherent observations. A stable
appearance in one optical image alone is not proof of stability. If necessary,
enlarge the processing AOI to include suitable reference ground.

The runner accepts `sentinel1.reference_area` as a file path relative to the
project configuration, `--reference-area /path/to/reference.gpkg` as an override,
or `sentinel1.reference_area_wkt`. Files are reprojected to EPSG:4326 and all
polygon features are combined; every component contributes to the reference mask. It fails
if the area is outside the AOI or has no usable phase in any pair. Without a
reference, displacement metadata explicitly says `spatial_reference=unreferenced`.

Run from the repository root; both project configurations now load the supplied references:

```bash
docker compose -f docker/docker-compose.yml exec -T -u "$(id -u):$(id -g)" gaiatesting \
  python3 -u -m subsystems.dpr.run_s1_eou_dpr \
  --project slope_monitoring_cadia --skip-download \
  --result-dir /data/gaia_tsf/slope_monitoring_cadia/sentinel1/results_corrected

docker compose -f docker/docker-compose.yml exec -T -u "$(id -u):$(id -g)" gaiatesting \
  python3 -u -m subsystems.dpr.run_s1_eou_dpr \
  --project slope_monitoring_jagersfontein --skip-download \
  --result-dir /data/gaia_tsf/slope_monitoring_jagersfontein/sentinel1/results_corrected
```

Supplied references have been copied into each project’s `static` directory:
Cadia uses `Stable_reference_Cadia_Gold_AUS.gpkg` (9 polygons, approximately
1.545 km² total); Jagersfontein uses `Stable_refernce_Jagersfontein_SAF.gpkg`
(3 polygons, approximately 0.112 km² total). Every feature is valid, in
EPSG:4326, covered by its processing AOI, and disjoint from its TSF mask.
These geometry checks do not establish radar coherence or physical stability;
those remain subject to processing and independent validation.
`--skip-download` processes existing local SLCs; it does not filter their dates.
Download searches now use the project's configured period and direction. Cadia
is configured for descending path 45; Jagersfontein retains ascending selection.
Use an isolated input directory for a strictly pre-failure rerun. Both current
project periods extend beyond failure, and full-period velocity is not a
pre-failure velocity estimate.

Inspect `quality/mean_coherence.tif`, `valid_pair_fraction.tif` and
`pair_rmse_mm.tif` alongside displacement. RMSE is internal fit consistency,
not absolute accuracy: a network with no redundant pairs can fit noise exactly.
Disconnected temporal dates are masked, but spatial SNAPHU component offsets
still require inspection. Compare all dates with the same diverging colour
scale in mm and extract failure-area and stable-area time series. Grayscale
screenshots without a scale or quality masks cannot settle the question.

## Validation

Fifteen numerical regressions pass in the installed processing container,
including broad deformation preservation, stable referencing, Boolean masking,
phase branch-cut averaging, temporal disconnection, velocity units, unwrapped
millimetre RMSE, coordinate alignment, and raster metadata. These tests exercise
the installed PyGMTSAR inversion and velocity code, but mock spatial unwrapping
and geocoding; they do not replace a full site rerun.

```bash
docker exec gaiatesting python3 -m pytest -q \
  subsystems/dpr/tests/test_sentinel1_regressions.py
```

## Cadia reference-phase failure: follow-up investigation

The 17 September corrected run stopped before export because all 12 pairs
involving **2017-10-04** had no usable data. This was not a destination-path
problem or an empty reference geometry.

Reconstruction from the retained aligned SLCs and radar transform found:

- The nine reference polygons cover **7,011 pixels** on the interferogram grid.
- Of 555 pairs, **543 have 5,708–6,842 reference pixels above coherence 0.20**.
- The other **12 pairs all contain 2017-10-04**. They have no coherent pixels
  anywhere in the processing grid and NaN maximum reference coherence.
- The aligned 2017-10-04 SLC contains no nonzero samples in the AOI. Its
  nonzero samples occupy radar rows 0–197, outside the AOI rows approximately
  1022–1559. Neighbouring dates have nonzero samples in rows 1030–1541.
- Only the later source burst is present locally for 2017-10-04, whereas
  2017-09-22 and 2017-10-16 each have two source bursts. The affected aligned
  SLC has 1,456 rows versus 2,800 rows for those neighbours, and an azimuth
  shift of −1348 versus −9. This implicates incomplete burst coverage and/or
  its handling during alignment. Determining why the earlier source burst is
  absent requires a separate archive/download audit; it is not established here.
- A direct SNAPHU check on the first pair (2015-03-01 / 2015-04-30) succeeded,
  producing 112,923 valid unwrapped pixels. The all-pair unwrapping diagnostic
  was stopped once the upstream empty-SLC cause was established.

Evidence: [per-pair coherence counts](diagnostics/cadia/reference_coherence_counts.csv)
and [per-acquisition SLC coverage](diagnostics/cadia/quality/slc_coverage.csv).
The diagnostic reconstruction reused aligned files and did not overwrite the
old or corrected displacement products.

The Cadia configuration now explicitly excludes **2017-10-04** as an unusable
acquisition. Removing its 12 edges leaves **103 dates and 543 pairs in one
connected network**. All January–March 2018 acquisitions remain available.
This is a workaround for the empty acquisition, not a repair of its SLC.
Source files are preserved and the coherence threshold and reference polygons
are unchanged. The same rerun command in this document can be used.

The pipeline now checks aligned SLC coverage before forming interferograms and
writes `quality/slc_coverage.csv`. A reference-phase failure also writes
`quality/reference_phase_diagnostics.csv` with per-pair coherent-reference,
unwrapped-reference and whole-image valid counts, and names affected pairs in
the error. Eighteen regression tests pass. A full corrected displacement run
has not been completed as part of this investigation.

## Automatically saved intermediate diagnostics

Diagnostics are enabled by default on subsequent runs; the existing command
needs no changes. Set `sentinel1.diagnostics: false` in project configuration
only when these additional reports are not wanted.

Each run writes a separate directory:

```text
<site data_dir>/diagnostics/<UTC-run-timestamp>/
  README.md
  summary.json
  network_candidates.csv / .json
  sbas_pairs.csv / .json
  acquisition_degree.csv / .json
  sbas_network.png
  slc_coverage.csv / .json
  wrapped_phase.csv / .json / .png
  wrapped_phase_sample_*.png
  unwrapped_phase.csv / .json / .png
  unwrapped_phase_sample_*.png
  reference_phase.csv / .json / .png
  displacement_by_date.csv / .json / .png
```

`summary.json` records running/completed/failed state, stage timings, errors,
configuration and library versions. Reports are written as stages complete,
including input coverage and wrapped-phase statistics before SNAPHU. A failed
run retains its reports, and later runs do not overwrite them. Reports live
outside the temporary work directory and survive normal cleanup.

The SBAS plot shows acquisition dates versus perpendicular baseline, with edges
for selected pairs; it is a temporal/baseline network rather than a geographic
map. Phase/coherence previews use radar coordinates. For each phase stage,
previews sample the first and least-covered pair, limiting plot overhead. The
reference plot displays combined usable pixel counts and per-pair mean phase
offsets. Whole-grid displacement plots are not dam-wall time series, and their
valid spatial support can change. Mean coherence in phase reports uses only
accepted pixels, so inspect coverage alongside it.

These diagnostics do not change filtering thresholds or the scientific solution.
They do trigger additional reductions and write PNG/CSV/JSON files. They are not
full phase checkpoints, spatial component validation, phase-closure tests,
atmospheric corrections, or independent uncertainty estimates. Those remain
separate improvements. The JSON summary's stage timing includes the work
actually evaluated there; lazy work may be evaluated in a later stage.

The project runner stores these run reports under the site data directory, independently
of `--result-dir`. For Cadia this is `/data/gaia_tsf/slope_monitoring_cadia/diagnostics/`
in Docker, mapped to `/home/lukas/GAIA-TSF/tsf_experiments/slope_monitoring_cadia/diagnostics/`
on the host. Direct pipeline callers can set `diagnostics_root`; without it the
previous result-directory default remains available. Quality GeoTIFFs and existing
quality CSVs continue to accompany the displacement products in `result_dir/quality/`.
