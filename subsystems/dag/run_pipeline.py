from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description='Run DAG pipelines.')
    parser.add_argument(
        '--pipeline',
        required=True,
        choices=['slope_eda', 'slope_features', 'slope_temporal_features'],
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

    if args.pipeline == 'slope_eda':
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

    else:
        raise ValueError(f'Unknown pipeline: {args.pipeline}')


if __name__ == '__main__':
    main()
