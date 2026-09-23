"""Configuration-driven DAG entry point for DA_R_01 and the MAP handoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description='Run DAG pipelines.')
    parser.add_argument(
        '--pipeline',
        required=True,
        choices=[
            'amd_index',
            'amd_eda',
            'amd_features',
            'amd_temporal_features',
            'amd_meteo_features',
            'slope_eda',
            'slope_features',
            'slope_temporal_features',
            'meteo_features',
            'topographic_features',
        ],
        help='Pipeline to run.',
    )
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help='Path to config.yaml.',
    )
    return parser


def main() -> None:
    """Run a configured DAG pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    if args.pipeline in ('amd_features', 'amd_temporal_features', 'amd_meteo_features'):
        from subsystems.dag.pipelines.amd_model_feature_pipeline import (
            AMDFeaturePipeline, AMDTemporalFeaturePipeline, AMDMeteoFeaturePipeline,
        )
        pipelines = {'amd_features': AMDFeaturePipeline,
                     'amd_temporal_features': AMDTemporalFeaturePipeline,
                     'amd_meteo_features': AMDMeteoFeaturePipeline}
        print(pipelines[args.pipeline](args.config).run())

    elif args.pipeline == 'amd_eda':
        from subsystems.dag.pipelines.amd_eda_pipeline import AMDEDAPipeline

        print(AMDEDAPipeline(args.config).run())

    elif args.pipeline == 'amd_index':
        from subsystems.dag.pipelines.amd_index_pipeline import AMDIndexPipeline

        print(AMDIndexPipeline(args.config).run())

    elif args.pipeline == 'slope_eda':
        from subsystems.dag.pipelines.slope_eda_pipeline import SlopeEDAPipeline

        result = SlopeEDAPipeline(args.config).run()
        print(len(result))

    elif args.pipeline == 'slope_features':
        from subsystems.dag.pipelines.slope_feature_pipeline import SlopeFeaturePipeline

        result = SlopeFeaturePipeline(args.config).run()
        print(len(result))

    elif args.pipeline == 'slope_temporal_features':
        from subsystems.dag.pipelines.slope_temporal_feature_pipeline import (
            SlopeTemporalFeaturePipeline,
        )

        result = SlopeTemporalFeaturePipeline(args.config).run()
        print(len(result))

    elif args.pipeline == 'meteo_features':
        from subsystems.dag.pipelines.meteo_feature_pipeline import (
            MeteoFeaturePipeline,
        )

        result = MeteoFeaturePipeline(args.config).run()
        print(len(result))

    elif args.pipeline == 'topographic_features':
        from subsystems.dag.pipelines.topographic_feature_pipeline import (
            TopographicFeaturePipeline,
        )

        result = TopographicFeaturePipeline(args.config).run()
        print(len(result))

    else:
        raise ValueError(f'Unknown pipeline: {args.pipeline}')


if __name__ == '__main__':
    main()
