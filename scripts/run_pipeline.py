#!/usr/bin/env python3
"""Project-level runner: MAP inference, then ALE (each also runnable alone).

Runs each subsystem through its own command line in a separate process, so
this script imports neither MAP nor ALE and the two never import each other:

1. MAP:  ``subsystems/map/run_inference.py --config <map config>``
2. ALE:  ``python -m ale_3levels run --engine <engine> --config <ale config> --out <out>``
3. optional report: ``python -m ale_3levels compare ... --truth <mask>``

Examples (repository root)::

    python3 scripts/run_pipeline.py --map-config MAP.yaml --ale-config ALE.yaml --out OUT
    python3 scripts/run_pipeline.py --skip-map --ale-config ALE.yaml --out OUT --truth MASK
    python3 scripts/run_pipeline.py --skip-ale --map-config MAP.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ALE_ROOT = REPOSITORY / 'subsystems' / 'ale'
MAP_INFERENCE = REPOSITORY / 'subsystems' / 'map' / 'run_inference.py'


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print('$', ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--map-config', type=Path, help='MAP config.yaml')
    parser.add_argument('--ale-config', type=Path, help='ALE configuration')
    parser.add_argument('--out', type=Path, help='ALE output directory')
    parser.add_argument(
        '--engine', choices=('fixed_gates', 'three_level', 'both'), default='both'
    )
    parser.add_argument('--model', help='MAP model driving the alerts')
    parser.add_argument('--calibration', type=Path, help='frozen engine-B calibration')
    parser.add_argument('--state', type=Path, help='previous engine-B state')
    parser.add_argument('--truth', type=Path, help='ground truth for the report')
    parser.add_argument(
        '--skip-map', action='store_true', help='use existing MAP output'
    )
    parser.add_argument('--skip-ale', action='store_true', help='run MAP only')
    args = parser.parse_args(argv)
    if not args.skip_map and args.map_config is None:
        parser.error('--map-config is required unless --skip-map is given.')
    if not args.skip_ale and (args.ale_config is None or args.out is None):
        parser.error('--ale-config and --out are required unless --skip-ale is given.')

    python = sys.executable
    if not args.skip_map:
        _run(
            [python, str(MAP_INFERENCE), '--config', str(args.map_config.resolve())],
            cwd=REPOSITORY,
        )
    if args.skip_ale:
        return 0
    env = {**os.environ, 'PYTHONPATH': str(ALE_ROOT)}
    out = args.out.resolve()
    command = [
        python,
        '-m',
        'ale_3levels',
        'run',
        '--engine',
        args.engine,
        '--config',
        str(args.ale_config.resolve()),
        '--out',
        str(out),
    ]
    for flag, value in (
        ('--model', args.model),
        ('--calibration', args.calibration),
        ('--state', args.state),
    ):
        if value is not None:
            command += [
                flag,
                str(Path(value).resolve()) if flag != '--model' else value,
            ]
    _run(command, cwd=ALE_ROOT, env=env)
    if args.truth is not None and args.engine == 'both':
        _run(
            [
                python,
                '-m',
                'ale_3levels',
                'compare',
                '--a',
                str(out / 'fixed_gates'),
                '--b',
                str(out / 'three_level'),
                '--truth',
                str(args.truth.resolve()),
                '--out',
                str(out / 'report'),
            ],
            cwd=ALE_ROOT,
            env=env,
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
