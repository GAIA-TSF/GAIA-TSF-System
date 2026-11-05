# SBAS network diagnostics

How to identify acquisitions that cannot connect to the rest of the SBAS network? 

1. Read the network diagnostics. network_candidates.csv records the tested temporal and perpendicular-baseline limits and the number of connected groups (n_comp). A connected network has n_comp = 1.

2. Recalculate baselines from the saved alignment metadata. Use the .PRM files to calculate each acquisition’s perpendicular baseline relative to the stack reference.

3. Reconstruct the network. Each acquisition date is a node. Two dates are linked when their time separation and perpendicular-baseline difference satisfy the thresholds.

4. Find disconnected groups and their nearest possible links. Western Platinum had one group containing 341 dates, plus two isolated dates:

   Isolated acquisition: 23 April 2015 156.6 m; 11 April 2024 358.1 m  Both exceeded the original 150 m limit.

5. Test the proposed exclusion before rerunning processing. Removing those two dates produced a connected network at 60 days / 150 m. 
   Alternatively, excluding only April 2024 and allowing 160 m retained the April 2015 acquisition.

An isolated acquisition is not necessarily a bad or corrupt image. It is unsuitable for the selected pairing limits. 


Run from the repository root in the processing container (PyGMTSAR and GMTSAR
are required to calculate baselines from PRM files):

```bash
python -m subsystems.dpr.scripts.diagnose_sbas_network \
  --workdir /data/gaia_tsf/slope_monitoring_western_platinum/processing/sentinel1/pre_failure \
  --output-dir /data/gaia_tsf/slope_monitoring_western_platinum/diagnostics/network_review \
  --days 120 --meters 150
```

Use an existing failed-run workdir before another pipeline run resets it.
Successful pipeline cleanup removes PRMs; preserve the exported `baselines.csv`
for subsequent analysis. `--reference-date YYYY-MM-DD` should match the original
alignment reference (default: earliest date). If several PRM subswath groups are
present, explicitly choose `--subswath`, e.g. `2` for `_F2.PRM`.

To test exclusions or thresholds using exported baselines:

```bash
python -m subsystems.dpr.scripts.diagnose_sbas_network \
  --baseline-csv /path/to/network_review/baselines.csv \
  --output-dir /path/to/network_trial \
  --days 60 --meters 150 \
  --exclude-dates 2015-04-23 2024-04-11
```

Exports:
- `network_report.json`: baseline provenance, thresholds, components, degrees,
  pair list, isolated dates, nearest spatial/temporal bridge options per component
  pair, and optional exclusion-trial results.
- `baselines.csv`: reusable date/baseline table.
- `sbas_network.png`: date-versus-baseline network coloured by component.
- `acquisition_degree.png`: connection counts by date.
- `exclusion_trial/`: plots after hypothetical exclusions, if requested.

Output filenames are replaced when the same output directory is reused.
The script does not alter sources or configuration and does not automatically
exclude acquisitions. It matches PyGMTSAR's spatial comparison
`abs(Bdiff) < meters + 1`, not a strict `<= meters`. A connected graph is not proof
of coherence or correct unwrapping. Baselines recalculated from another reference
can differ slightly from an earlier run. The diagnostic covers all dates in the
selected PRM group; it does not read epoch configuration or scan raw SAFE files.
