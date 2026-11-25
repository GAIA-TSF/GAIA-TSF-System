#!/usr/bin/env bash
# Restore the synthetic site and its MAP inference outputs for ALE.
#
# 1. Copy the synthetic external-trigger site (inputs, static, DAG features,
#    trained slope_trf model) from its source experiment directory into
#    $ALE_DATA_DIR/synthetic_site, outside the repository. The source is only read.
# 2. Run MAP inference on the copy with MAP's own config, changing only
#    datasets.slope_dataset.experiment_dir.
# 3. Validate the MAP residual table with ALE's contract reader.
#
# Usage: subsystems/ale/scripts/prepare_synthetic.sh [--force]
#   SYNTHETIC_SOURCE   source experiment directory (default below)
#   ALE_DATA_DIR       ALE data directory (default below; config/ale_synthetic.yaml
#                      points to the same default location)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ALE="$ROOT/subsystems/ale"
WORK="${ALE_DATA_DIR:-$HOME/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV}"
SITE="$WORK/synthetic_site"
SOURCE="${SYNTHETIC_SOURCE:-$HOME/ownCloud/Projects/GAIA_TSF/tsf_experiments/slope_synthetic_western_platinum_external_trigger}"
EXPECTED_MODEL_SHA="51fd24bbbf12"   # slope_trf@51fd24bbbf12 (golden files)

if [[ ! -d "$SOURCE/results/models/slope_trf" ]]; then
  echo "Synthetic site source not found: $SOURCE (set SYNTHETIC_SOURCE)" >&2
  exit 1
fi
if [[ -d "$SITE" && "${1:-}" != "--force" ]]; then
  echo "$SITE exists; use --force to rebuild it." >&2
  exit 1
fi
model_sha="$(sha256sum "$SOURCE/results/models/slope_trf/model.pkl" | cut -c1-12)"
if [[ "$model_sha" != "$EXPECTED_MODEL_SHA" ]]; then
  echo "Warning: model.pkl sha256 $model_sha differs from $EXPECTED_MODEL_SHA;" \
       "results will not match the documented baseline." >&2
fi

case "$(cd "$(dirname "$WORK")" 2>/dev/null && pwd)/$(basename "$WORK")" in
  "$ROOT"|"$ROOT"/*) echo "ALE_DATA_DIR must be outside the repository: $WORK" >&2; exit 1 ;;
esac
mkdir -p "$WORK"

echo "== copying site from $SOURCE"
rm -rf "$SITE"
mkdir -p "$SITE/results"
cp -r "$SOURCE/inputs" "$SOURCE/static" "$SOURCE/scenario_config.yaml" "$SITE/"
cp -r "$SOURCE/results/temporal_features" "$SOURCE/results/models" "$SITE/results/"
[[ -d "$SOURCE/results/features" ]] && cp -r "$SOURCE/results/features" "$SITE/results/"

echo "== MAP inference"
python3 - "$ROOT/subsystems/map/config.yaml" "$WORK/map_config_synthetic.yaml" "$SITE" <<'EOF'
import sys
import yaml
source, target, site = sys.argv[1:]
config = yaml.safe_load(open(source, encoding='utf-8'))
config['dataset'] = 'slope_dataset'
config['model'] = 'trf'
config['datasets']['slope_dataset']['experiment_dir'] = site
with open(target, 'w', encoding='utf-8') as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
EOF
(cd "$ROOT" && python3 subsystems/map/run_inference.py --config "$WORK/map_config_synthetic.yaml" \
  2>&1 | grep -v "InconsistentVersionWarning\|warnings.warn\|scikit-learn.org")

echo "== ALE contract validation"
(cd "$ALE" && python3 -m ale_3levels validate --config config/ale_synthetic.yaml \
  > "$WORK/synthetic_contract_summary.json")
python3 - "$WORK/synthetic_contract_summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
split = s['records_by_split']
print(f"model {s['model_version']}  units {s['declared_units']} -> {s['source_units']}  "
      f"dates {s['dates']['count']} ({s['dates']['first']} .. {s['dates']['last']})")
print(f"calibration-split dates {s['calibration_split_dates']}  monitoring start {s['monitoring_start']}  "
      f"records: {split}")
EOF
echo "== done: $SITE"
