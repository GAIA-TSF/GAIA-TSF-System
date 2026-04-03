import sys
from pathlib import Path
import argparse
import yaml

# Ensure project root in path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subsystems.dag.core.executor import PipelineExecutor
from subsystems.dag.pipelines.amd_pipeline import AMDPipeline
from subsystems.dag.pipelines.slope_pipeline import SlopeStabilityPipeline


# Config loader
def load_config(path: str):
    print(f'[DEBUG] Loading config: {path}')

    with open(path, 'r') as f:
        return yaml.safe_load(f)


# Build inputs from config
def build_inputs(config: dict, pipeline_name: str):
    print(f'[DEBUG] Building inputs for pipeline: {pipeline_name}')

    inputs_cfg = config.get('inputs', {})

    if pipeline_name == 'amd':
        amd_cfg = inputs_cfg.get('amd', {})
        return {
            's2': [amd_cfg.get('sentinel2', {}).get('data_path')],
            'aoi': amd_cfg.get('auxiliary', {}).get('aoi_path'),
            'water_mask': amd_cfg.get('auxiliary', {}).get('water_mask_path'),
            'insitu': amd_cfg.get('auxiliary', {}).get('insitu_path'),  # optional
        }

    elif pipeline_name == 'slope':
        slope_cfg = inputs_cfg.get('slope_stability', {})
        return {
            's1': [slope_cfg.get('sentinel1', {}).get('data_path')],
            'aoi': slope_cfg.get('auxiliary', {}).get('aoi_path'),
        }

    else:
        raise ValueError(f'Unknown pipeline: {pipeline_name}')


# Pipeline factory
def get_pipeline(name: str):
    if name == 'amd':
        return AMDPipeline()
    elif name == 'slope':
        return SlopeStabilityPipeline()
    else:
        raise ValueError(f'Unknown pipeline: {name}')


# MAIN
def main():
    parser = argparse.ArgumentParser(description='DAG Debug Runner')

    parser.add_argument(
        '--config',
        required=True,
        help='Path to config.yaml',
    )

    parser.add_argument(
        '--pipeline',
        required=True,
        choices=['amd', 'slope'],
        help='Pipeline to run',
    )

    args = parser.parse_args()

    print('[DAG DEBUG RUNNER]')
    print(f'[DEBUG] Pipeline: {args.pipeline}')

    config = load_config(args.config)

    inputs = build_inputs(config, args.pipeline)

    executor = PipelineExecutor()
    executor.register(args.pipeline, get_pipeline(args.pipeline))

    result = executor.run(args.pipeline, inputs)

    print('\n[RESULT]')
    print(result)


if __name__ == '__main__':
    main()
