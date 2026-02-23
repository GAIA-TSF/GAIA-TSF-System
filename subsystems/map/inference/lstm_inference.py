
from subsystems.map.inference.pipeline import run_lstm_experiment
import argparse

""" 
Entry point only. 
"""

def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run LSTM monitoring experiment"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["synthetic", "mirmazloumi_2023"],
    )

    parser.add_argument(
        "--config",
        default="subsystems/map/learning/config.yaml",
    )

    return parser.parse_args()


def main():
    args = _parse_arguments()
    run_lstm_experiment(args.dataset, args.config)


if __name__ == "__main__":
    main()