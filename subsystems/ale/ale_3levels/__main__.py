"""Command line: ``python -m ale_3levels {validate,run,calibrate,compare}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ale_3levels.compare import compare as compare_engines
from ale_3levels.config import DEFAULT_CONFIG, load_config
from ale_3levels.contract import ContractError, summary
from ale_3levels.engines.three_level.calibration import CalibrationError
from ale_3levels.run import load_input
from ale_3levels.run import run as run_engines


def _contract_from_config(input_path: Path | None, config_path: Path):
    return load_input(load_config(config_path), input_path)


def _not_yet(step: str) -> int:
    print(f'Not implemented yet ({step} in ALE_PLAN.md §7).', file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m ale_3levels', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    validate = commands.add_parser(
        'validate', help='check a MAP table against the contract'
    )
    validate.add_argument('--input', type=Path, help='MAP residuals.npz')
    validate.add_argument('--config', type=Path, default=DEFAULT_CONFIG)

    run = commands.add_parser('run', help='score MAP output with one or both engines')
    run.add_argument(
        '--engine', choices=('fixed_gates', 'three_level', 'both'), default='both'
    )
    run.add_argument('--input', type=Path)
    run.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument(
        '--calibration',
        type=Path,
        help="frozen engine-B calibration directory (from 'calibrate')",
    )

    calibrate = commands.add_parser('calibrate', help='freeze engine calibration')
    calibrate.add_argument('--engine', choices=('three_level',), default='three_level')
    calibrate.add_argument('--input', type=Path)
    calibrate.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    calibrate.add_argument('--out', type=Path, required=True)

    compare = commands.add_parser(
        'compare', help='compare two engine output directories'
    )
    compare.add_argument('--a', type=Path, required=True)
    compare.add_argument('--b', type=Path, required=True)
    compare.add_argument('--out', type=Path, required=True)
    compare.add_argument(
        '--truth', type=Path, help='ground-truth failure mask (report only)'
    )
    compare.add_argument('--title', default='ALE engine comparison')

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(name)s: %(message)s'
    )
    try:
        if args.command == 'validate':
            contract = _contract_from_config(args.input, args.config)
            print(json.dumps(summary(contract), indent=2))
            return 0
        if args.command == 'run':
            written = run_engines(
                args.engine,
                args.out,
                config_path=args.config,
                input_path=args.input,
                calibration_dir=args.calibration,
            )
            for engine, paths in written.items():
                print(f'{engine}: {paths["alerts"].parent}')
            return 0
        if args.command == 'calibrate':
            return _not_yet('step 8')
        paths = compare_engines(
            args.a, args.b, args.out, truth=args.truth, title=args.title
        )
        print(paths['report'])
        return 0
    except ContractError as error:
        print(f'Contract error: {error}', file=sys.stderr)
        return 1
    except CalibrationError as error:
        print(f'Calibration error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
