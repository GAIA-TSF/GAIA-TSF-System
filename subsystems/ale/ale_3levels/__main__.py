"""Command line: ``python -m ale_3levels {validate,run,calibrate,compare}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ale_3levels.config import DEFAULT_CONFIG, load_config
from ale_3levels.contract import ContractError, load_contract, read_mask, summary


def _contract_from_config(input_path: Path | None, config_path: Path):
    config = load_config(config_path)
    source = config.get('input') or {}
    site = config.get('site') or {}
    path = input_path or source.get('residuals')
    if path is None:
        raise ContractError('No MAP table given (--input or input.residuals).')
    mask_file = source.get('mask_file')
    return load_contract(
        Path(path),
        lead=int(source.get('lead', 1)),
        model_id=source.get('model_id'),
        source_units_override=source.get('source_units_override'),
        mask=read_mask(Path(mask_file)) if mask_file else None,
        site_id=site.get('site_id'),
        los_sign_convention=site.get(
            'los_sign_convention', 'negative_away_from_satellite'
        ),
        reference_date=site.get('reference_date'),
        reference_point=site.get('reference_point'),
    )


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
            return _not_yet('steps 5-6')
        if args.command == 'calibrate':
            return _not_yet('step 8')
        return _not_yet('step 9')
    except ContractError as error:
        print(f'Contract error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
