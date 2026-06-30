"""
CLI entry point for training

Usage:
    python -m subsystems.map.run_learning --config config.yaml

Purpose:
- Load config from YAML (user-specified)
- Register plugins
- Run unified learning pipeline
"""

import sys
import os
import argparse


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

print(f'[DEBUG] Using ROOT_DIR: {ROOT_DIR}')


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

    # print(f"[Pipeline] Loaded config: {config}")

    # 3. Register plugins

    # 4. Run pipeline
    from pipelines.learning_pipeline import run_learning

    run_learning(config)


if __name__ == '__main__':
    main()
