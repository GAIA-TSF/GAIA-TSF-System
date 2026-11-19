"""Compare MAP models and the constant baseline on identical observations.

Example:
    python3 subsystems/map/run_compare.py \
        trf=<experiment_dir>/results/forecasts/slope_trf/residuals.npz \
        lstm=<experiment_dir>/results/forecasts/slope_lstm/residuals.npz \
        --roles validation test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.map.utils.forecasts import compare_residual_tables, read_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('tables', nargs='+', help='name=path/to/residuals.npz')
    parser.add_argument('--roles', nargs='+', help='restrict to sample_role values')
    parser.add_argument('--period', choices=('calibration', 'monitoring', 'outside'))
    args = parser.parse_args()
    tables = {}
    for item in args.tables:
        name, _, path = item.partition('=')
        if not path:
            parser.error(f'Expected name=path, got {item!r}')
        tables[name] = read_table(Path(path))
    roles = None if args.roles is None else tuple(args.roles)
    print(json.dumps(compare_residual_tables(tables, roles, args.period), indent=2))


if __name__ == '__main__':
    main()
