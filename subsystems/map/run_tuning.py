"""Command-line entry point: tune TRF or LSTM settings, then retrain."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.map.pipelines.tuning_pipeline import run_tuning
from subsystems.map.utils.config_loader import load_config


def main() -> None:
    """Parse the configuration and optional selections, then tune."""
    parser = argparse.ArgumentParser(
        description='Tune TRF or LSTM by rolling-origin cross-validation.'
    )
    parser.add_argument('--config', type=Path, required=True, help='MAP config.yaml')
    parser.add_argument('--dataset', help='datasets.<name> to use (default: config dataset)')
    parser.add_argument('--model', choices=('trf', 'lstm'), help='override config model')
    parser.add_argument('--experiment', help='experiment name (default: <kv>_<model>)')
    parser.add_argument(
        '--no-retrain', action='store_true', help='only search; do not retrain the best model'
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    config = load_config(args.config, args.dataset, args.model, args.experiment)
    run_tuning(config, retrain=not args.no_retrain)


if __name__ == '__main__':
    main()
