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
from ale_3levels.run import ENGINE_CHOICES, load_input
from ale_3levels.run import calibrate as calibrate_engine
from ale_3levels.run import run as run_engines
from ale_3levels.state import StateMismatchError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m ale_3levels', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    validate = commands.add_parser(
        'validate', help='check a MAP table against the contract'
    )
    validate.add_argument('--input', type=Path, help='MAP residuals.npz')
    validate.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    validate.add_argument(
        '--model', help='MAP model driving the alerts (default input.model_id)'
    )

    run = commands.add_parser('run', help='score MAP output with one or both engines')
    run.add_argument(
        '--engine',
        choices=ENGINE_CHOICES,
        default='both',
        help='A = fixed_gates, B = three_level, or both (default)',
    )
    run.add_argument('--input', type=Path)
    run.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    run.add_argument(
        '--model', help='MAP model driving the alerts (default input.model_id)'
    )
    run.add_argument('--out', type=Path, required=True)
    run.add_argument(
        '--calibration',
        type=Path,
        help="frozen engine-B calibration directory (from 'calibrate')",
    )
    run.add_argument(
        '--state',
        type=Path,
        help='previous engine-B state directory (<previous out>/three_level/state)',
    )

    calibrate = commands.add_parser('calibrate', help='freeze engine calibration')
    calibrate.add_argument('--engine', choices=('three_level',), default='three_level')
    calibrate.add_argument('--input', type=Path)
    calibrate.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    calibrate.add_argument('--out', type=Path, required=True)
    calibrate.add_argument('--model', help='MAP model (default input.model_id)')

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

    plot = commands.add_parser('plot', help='plots from a saved run (both engines)')
    plot.add_argument(
        '--run', type=Path, required=True, help="'run --engine both' output"
    )
    plot.add_argument('--out', type=Path, required=True)
    plot.add_argument(
        '--no-maps', action='store_true', help='skip per-acquisition maps'
    )
    plot.add_argument('--animation', action='store_true', help='also write alerts.gif')

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(name)s: %(message)s'
    )
    try:
        if args.command == 'validate':
            contract = load_input(load_config(args.config), args.input, args.model)
            print(json.dumps(summary(contract), indent=2))
            return 0
        if args.command == 'run':
            written = run_engines(
                args.engine,
                args.out,
                config_path=args.config,
                input_path=args.input,
                calibration_dir=args.calibration,
                model_id=args.model,
                state_dir=args.state,
            )
            for engine, paths in written.items():
                print(f'{engine}: {paths["alerts"].parent}')
            return 0
        if args.command == 'calibrate':
            root = calibrate_engine(
                args.out,
                config_path=args.config,
                input_path=args.input,
                model_id=args.model,
            )
            print(root)
            return 0
        if args.command == 'plot':
            from ale_3levels.plots import plot_run

            written = plot_run(
                args.run, args.out, maps=not args.no_maps, animation=args.animation
            )
            print(f'{len(written)} files in {args.out}')
            return 0
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
    except StateMismatchError as error:
        print(f'State error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
