# ALE demonstration: engine A and engine B on a simulated TSF

This guide runs both ALE engines on a simulated tailings storage facility (TSF) and explains what to show. Nothing is real: the "MAP output" is simulated and written outside the repository.

- **Engine A, `fixed_gates` (option `A`):** looks at the absolute LOS movement over 30 days and gives WHITE, GREEN, YELLOW or RED.
- **Engine B, `three_level` (option `B`):** looks at how far each observation is from the MAP forecast, and gives L1 (single-date outlier), L2 (sustained deviation) or L3 (regime change).

## Run it

From the repository root:

```bash
python3 subsystems/ale/scripts/run_demo.py               # both engines, report, plots, GIF (~1 min)
python3 subsystems/ale/scripts/run_demo.py --engine A    # engine A only
python3 subsystems/ale/scripts/run_demo.py --engine B    # engine B only
python3 subsystems/ale/scripts/run_demo.py --new-site    # regenerate the simulated site
python3 subsystems/ale/scripts/run_demo.py --quick --no-plots   # small site, seconds
```

Output goes to `$ALE_DATA_DIR/showcase`. By default that is `~/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV/showcase`; `--out DIR` sets another directory, which must be outside the repository. Each call replaces the earlier runs and keeps the site. The same engines also run through the normal command line:

```bash
cd subsystems/ale
python3 -m ale_3levels run --engine A --config <site>/ale_showcase.yaml --out <dir>
python3 -m ale_3levels run --engine B --config <site>/ale_showcase.yaml --out <dir>
```

| Output | What it is |
|---|---|
| `zone_summary.md` | First date each zone reaches each level, per engine; also printed on screen |
| `report/report.md` | Comparison report: failure area against a stable control area, lead times, flip rates |
| `plots/zone_timeline.png` | Per zone over time: mean displacement, and engine A's and engine B's zone level, with monitoring start, failure onset and the atmospheric artefact marked |
| `plots/alerts.gif` | Engine A and engine B alert maps side by side, one frame per acquisition |
| `plots/maps/alerts_<date>.png` | The same maps as single images |
| `plots/three_level_history.png` | Engine B's TSF-wide L2 CUSUM and L3 score over time, with thresholds |
| `plots/sector_timeline.png` | Sector level over time for both engines |
| `runs/fixed_gates`, `runs/three_level` | Engine outputs (`alerts.npz`, `history.csv`, `sectors.csv`, ...) |

## The simulated site

An elliptical TSF on a 60 × 80 grid of 10 m pixels. It has 170 Sentinel-1-like acquisitions every 12 days, with one missed acquisition:

- calibration from 2019-01-05 to 2022-08-17;
- monitoring from 2022-08-29 to 2024-08-06, 60 dates.

| Zone | Pixels | Motion | Forecast by MAP? |
|---|---|---|---|
| stable | 2752 | slow uniform settlement, 0.6 mm per 30 days | yes |
| creep (west beach) | 234 | steady consolidation, 7.5 mm per 30 days | yes |
| failure (dam wall) | 130 | stable, then accelerating from 2023-01-20 towards a projected failure 30 days after the last acquisition (inverse-velocity law), up to about 4 mm/day | no |

- **Noise:** every acquisition carries 1 mm pixel noise and a smooth random atmospheric field of about 1.5 mm. One monitoring date, 2022-11-09, carries an extra strong atmospheric artefact of 8 mm.
- **Simulated MAP forecast:** persistence plus each pixel's learned rate. It predicts the stable and creep zones, but not the acceleration.
- **Sectors:** `west_beach`, `east_beach` and `dam_wall`.
- **Files for the presenter and the report only:**
  - `static/demo_zones.tif` holds the zones.
  - `static/roi_failure_mask.tif` is the ground truth.

  No engine reads either file.

## What to show

The results below come from the default site (seed 11). A zone "holds" a level when enough coherent pixels are at that level or higher, which is the same sector rule the report uses.

| Zone | Engine A first YELLOW / RED | Engine B first L1 / L2 / L3 |
|---|---|---|
| failure (from 2023-01-20) | 2023-03-09 / 2023-10-11 | 2023-07-31 / 2024-02-08 / 2024-08-06 |
| creep | YELLOW on 58 of 60 dates, from 2022-09-22 | L1 and L2 on 5 dates, at the very end |
| stable | GREEN on 54 of 60 dates; YELLOW once | L1 on 6 dates, L2 on 5 dates, at the very end |

1. **Engine A sees motion; engine B sees surprise.**
   - The creep zone moves steadily and as expected. Engine A keeps it YELLOW all the time, because it moves more than 4 mm per 30 days.
   - Engine B stays quiet there, because MAP predicts that motion. Show `plots/alerts.gif`: the west beach is YELLOW on the left and white on the right.
2. **Engine A is faster on the accelerating dam wall.** Show `plots/zone_timeline.png`: the dam wall's displacement curve bends down after the failure begins, and the two engines' zone levels step up at different times.
   - Engine A reaches YELLOW about 7 weeks after the acceleration begins, and RED about 9 months after.
   - Engine B escalates step by step: L1, then L2, and L3 only near the end. Show `plots/three_level_history.png`: the L3 score crosses watch (0.4) in May 2024 and alert (0.7) in July 2024.
3. **What each engine costs in false alerts.**
   - Engine A's GREEN is mostly noise. The stable zone is GREEN on almost every date, while YELLOW and RED stay clean.
   - Engine B's L2 is a TSF-wide alarm by design, so near the end the whole TSF shows L2 (yellow in the maps). L3 stays on the dam wall only.
4. **The atmospheric artefact on 2022-11-09 does not cause a lasting alarm** in either engine. Engine B was calibrated with atmospheric noise on every date, so one strong date does not latch its CUSUM.

In `report/report.md`, the "Ground truth" table shows the same first-alert dates for the failure area. Its "dates flagged" counts can differ by a few dates from `zone_summary.md`, because the report considers only its two areas, failure and control, when it checks which pixels form coherent groups.

## Notes

- **Reproducibility:** the simulation is deterministic, so a regenerated site gives the same numbers. Parameters are in `ale_3levels.demo.simulate_showcase`.
- **Configuration:** the engines use the default configuration. The site config (`ale_showcase.yaml`) only adds the input paths and the sectors file.
- **Small test site:** `scripts/make_demo_data.py` writes a much smaller site used by the tests. It is not meant for demonstrations.
