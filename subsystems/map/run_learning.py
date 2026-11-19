"""Command-line entry point: train and validate one MAP forecasting model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.map.pipelines.learning_pipeline import run_learning
from subsystems.map.utils.config_loader import load_config


def main() -> None:
    """Parse the configuration and optional selections, then run."""
    parser = argparse.ArgumentParser(description='Train and validate one MAP forecasting model.')
    parser.add_argument('--config', type=Path, required=True, help='MAP config.yaml')
    parser.add_argument('--dataset', help='datasets.<name> to use (default: config dataset)')
    parser.add_argument('--model', choices=('constant', 'trf', 'lstm'), help='override config model')
    parser.add_argument('--experiment', help='experiment name (default: <kv>_<model>)')
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    run_learning(load_config(args.config, args.dataset, args.model, args.experiment))


if __name__ == '__main__':
    main()
