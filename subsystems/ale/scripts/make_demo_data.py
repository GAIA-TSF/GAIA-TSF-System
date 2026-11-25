#!/usr/bin/env python3
"""Generate a small simulated ALE demo site (seconds, no MAP, no site data).

Writes a MAP-format residual table, TSF and ground-truth masks, an ALE config
and a README into ``--out`` (outside the repository). With ``--run`` it also
runs both engines and the comparison report into ``<out>/runs`` and
``<out>/report``.

Usage (from the repository root):
    python3 subsystems/ale/scripts/make_demo_data.py [--out DIR] [--run] [--force]

Default ``--out``: ``$ALE_DATA_DIR/demo``, where ``ALE_DATA_DIR`` defaults to
``~/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ALE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ALE_ROOT.parents[1]
sys.path.insert(0, str(ALE_ROOT))

from ale_3levels.compare import compare
from ale_3levels.demo import write_demo_site
from ale_3levels.run import run

DEFAULT_DATA_DIR = Path('~/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV')


def main(argv: list[str] | None = None) -> int:
    data_dir = Path(os.environ.get('ALE_DATA_DIR', DEFAULT_DATA_DIR)).expanduser()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--out', type=Path, default=data_dir / 'demo')
    parser.add_argument('--size', type=int, default=24, help='grid size in pixels')
    parser.add_argument('--dates', type=int, default=60, help='number of acquisitions')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--run', action='store_true', help='also run both engines and compare'
    )
    parser.add_argument(
        '--force', action='store_true', help='replace an existing --out'
    )
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    if out == REPOSITORY or REPOSITORY in out.parents:
        parser.error(f'--out must be outside the repository: {out}')
    if out.exists() and any(out.iterdir()):
        if not args.force:
            parser.error(f'{out} is not empty; use --force to replace it.')
        if not (out / 'ale_demo.yaml').is_file():
            parser.error(f'{out} does not look like a demo site; not removing it.')
        shutil.rmtree(out)

    monitoring = args.dates // 2
    paths = write_demo_site(
        out,
        size=args.size,
        n_dates=args.dates,
        monitoring_index=monitoring,
        failure_start=monitoring + 2,
        seed=args.seed,
    )
    print(f'demo site written: {out}')
    for name, path in paths.items():
        print(f'  {name}: {path}')
    if args.run:
        written = run('both', out / 'runs' / 'demo', config_path=paths['config'])
        report = compare(
            written['fixed_gates']['alerts'].parent,
            written['three_level']['alerts'].parent,
            out / 'report',
            truth=paths['truth_mask'],
            title='ALE demo site (simulated)',
        )
        print(f'report: {report["report"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
