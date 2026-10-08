"""Command-line entry point for MAP baseline learning."""

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
    """Parse the configuration path and run learning."""
    parser = argparse.ArgumentParser(description='Run the MAP learning pipeline.')
    parser.add_argument(
        '--config', type=Path, required=True, help='Path to MAP config.yaml'
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    run_learning(load_config(args.config))


if __name__ == '__main__':
    main()
