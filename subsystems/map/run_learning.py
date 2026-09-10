"""
CLI entry point for training

Usage:
    python -m subsystems.map.run_learning --config config.yaml

Purpose:
- Load config from YAML (user-specified)
- Register plugins
- Run unified learning pipeline
"""

import argparse
import os
import sys

from plugins.models.gbr import GBRModel  # noqa: F401
from plugins.models.lstm import LSTMModel  # noqa: F401
from plugins.models.rf import RandomForestModel  # noqa: F401
from plugins.models.xgb import XGBoostModel  # noqa: F401
from plugins.variables.amd import AMDVariable  # noqa: F401
from plugins.variables.slope import SlopeVariable  # noqa: F401


# Ensure correct ROOT_DIR (important for imports)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = CURRENT_DIR
while ROOT_DIR != '/':
    if (
        os.path.exists(os.path.join(ROOT_DIR, 'plugins'))
        and os.path.exists(os.path.join(ROOT_DIR, 'core'))
        and os.path.exists(os.path.join(ROOT_DIR, 'pipelines'))
    ):
        break
    ROOT_DIR = os.path.dirname(ROOT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# --------------------------------------------------
# CLI ARGUMENTS
# --------------------------------------------------
def parse_args():
    """
    Parse command line arguments.

    --config:
        Path to YAML configuration file
    """
    parser = argparse.ArgumentParser(description='Run learning pipeline')

    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')

    return parser.parse_args()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    # 1. Parse CLI args
    args = parse_args()

    # 2. Load config
    from utils.config_loader import load_config

    config = load_config(args.config)

    # 3. Register plugins

    # 4. Run pipeline
    from pipelines.learning_pipeline import run_learning

    run_learning(config)


if __name__ == '__main__':
    main()
