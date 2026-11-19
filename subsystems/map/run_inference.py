"""Command-line entry point: issue MAP forecasts and match available observations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.map.pipelines.inference_pipeline import run_inference
from subsystems.map.utils.config_loader import load_config


def main() -> None:
    """Parse the configuration and optional selections, then run."""
    parser = argparse.ArgumentParser(description='Issue MAP forecasts and match available observations.')
    parser.add_argument('--config', type=Path, required=True, help='MAP config.yaml')
    parser.add_argument('--dataset', help='datasets.<name> to use (default: config dataset)')
    parser.add_argument('--model', choices=('constant', 'trf', 'lstm'), help='override config model')
    parser.add_argument('--experiment', help='experiment name (default: <kv>_<model>)')
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    run_inference(load_config(args.config, args.dataset, args.model, args.experiment))


if __name__ == '__main__':
    main()
