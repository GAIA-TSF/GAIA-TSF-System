
"""
CLI entry point for inference
"""
# at top of run_learning.py
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
# Register plugins
import plugins.variables.amd
import plugins.features.temporal
import subsystems.map.plugins.models.gbr

from utils.config_loader import load_config
from pipelines.inference_pipeline import run_inference


if __name__ == "__main__":
    # Load config from YAML
    config = load_config("config.yaml")

    run_inference(config) 
