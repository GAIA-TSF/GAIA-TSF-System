"""
CLI entry point for inference
"""

from __future__ import annotations

import argparse
import os
import sys

from plugins.models.gbr import GBRModel  # noqa: F401
from plugins.models.lstm import LSTMModel  # noqa: F401
from plugins.models.rf import RandomForestModel  # noqa: F401
from plugins.models.xgb import XGBoostModel  # noqa: F401
from plugins.variables.amd import AMDVariable  # noqa: F401
from plugins.variables.slope import SlopeVariable  # noqa: F401

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipelines.inference_pipeline import run_inference
from utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MAP inference pipeline")
    parser.add_argument("--config", type=str, default="subsystems/map/config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    run_inference(config)
