import os 
import argparse
import yaml
import itertools
import json
import numpy as np

from .validate_lstm import run_validation
from .trainer import Trainer

# experiment folder
def _create_experiment_dir(cfg):

    root = cfg["experiments"]["root_dir"]
    name = cfg["experiments"]["name"]

    exp_dir = os.path.join(root, name)

    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir

# -------------------------------------------------
# argument parsing
# -------------------------------------------------
def _parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run LSTM hyperparameter tuning"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["synthetic", "mirmazloumi_2023"],
        help="Dataset type",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="subsystems/map/learning/config.yaml",
        help="Path to config file",
    )

    return parser.parse_args()


# -------------------------------------------------
# config loader
# -------------------------------------------------
def _load_config(path):

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# grid search
# -------------------------------------------------
def grid_search(dataset_name, config_path, tuning_cfg):

    cfg = _load_config(config_path)

    exp_dir = os.path.join(
        cfg["experiments"]["root_dir"],
        cfg["experiments"]["name"]
    )

    os.makedirs(exp_dir, exist_ok=True)

    param_names = []
    param_values = []

    for key, value in tuning_cfg.items():
        if isinstance(value, list):
            param_names.append(key)
            param_values.append(value)

    n_models = np.prod([len(v) for v in param_values])
    print("Total models to test:", n_models)

    best_loss = float("inf")
    best_params = None

    results = []

    for combo in itertools.product(*param_values):

        params = dict(zip(param_names, combo))

        print("\nTesting configuration")
        print(params)

        fold_losses = run_validation(
            dataset_name,
            config_path,
            override_params=params,
            save_results=False, 
        )

        mean_loss = np.mean(fold_losses)

        results.append(
            {
                "params": params,
                "mean_loss": float(mean_loss),
                "fold_losses": [float(x) for x in fold_losses],
            }
        )

        if mean_loss < best_loss:

            best_loss = mean_loss
            best_params = params
            model_path = os.path.join(exp_dir, cfg["experiments"]["model_file"])


    print("\nBest configuration")
    print(best_params)
    print("Loss:", best_loss)
    best_params_path = os.path.join(exp_dir, "best_params.json")

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    results.append(
        {
            "params": params,
            "mean_loss": float(mean_loss),
            "fold_losses": [float(x) for x in fold_losses],
        }
    )

    return best_params, results


# -------------------------------------------------
# main
# -------------------------------------------------
def main():

    args = _parse_arguments()

    cfg = _load_config(args.config)

    if "tuning" not in cfg:
        raise RuntimeError("No tuning section found in config")

    tuning_cfg = cfg["tuning"]

    best_params, results = grid_search(
        dataset_name=args.dataset,
        config_path=args.config,
        tuning_cfg=tuning_cfg,
    )

    # save results
    exp_dir = _create_experiment_dir(cfg)
    config_copy = os.path.join(exp_dir, "config_used.yaml")

    with open(config_copy, "w") as f:
        yaml.dump(cfg, f)
    
    output = {
        "best_params": best_params,
        "results": results,
    }

    output_path = os.path.join(exp_dir, "tuning_results.json")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nTuning results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
    