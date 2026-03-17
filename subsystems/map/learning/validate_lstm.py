import os 
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from ..dataset.insar import (
    create_synthetic_insar_dataset,
    create_mirmazloumi_2023_dataset,
)

from ..learning import LearningModule
from .validation import expanding_window_splits

import argparse

# config loading 
def _load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    

def _parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run LSTM time-series validation"
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
    
    
def run_validation(dataset_name, config_path, override_params=None, save_results=True):

    # print('Validation!')
    # with open(config_path) as f:
    #     cfg = yaml.safe_load(f)
    
    cfg = _load_config(config_path)

    exp_dir = os.path.join(
        cfg["experiments"]["root_dir"],
        cfg["experiments"]["name"]
    )

    os.makedirs(exp_dir, exist_ok=True)
    print("Validation experiment directory:", exp_dir) 

    trainer_cfg = cfg["trainer"]
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    # validation_cfg = cfg["validation"] TODO: check if to use it? 

    look_back = trainer_cfg["look_back"]
    horizon = trainer_cfg["horizon"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if override_params:

        for k, v in override_params.items():

            if k in cfg["model"]:
                cfg["model"][k] = v

            if k in cfg["trainer"]:
                cfg["trainer"][k] = v


    # ---------------- dataset ----------------
    if dataset_name == "synthetic":
        
        dataset = create_synthetic_insar_dataset(
            length=dataset_cfg["length"],
            noise_std=dataset_cfg["noise_std"],
            trend_amplitude=dataset_cfg["trend_amplitude"],
            anomaly_magnitude=dataset_cfg["anomaly_magnitude"],
            look_back=look_back,
            horizon=horizon,
        )
    else:
        dataset = create_mirmazloumi_2023_dataset(
            look_back=look_back,
            horizon=horizon,
        )

    n = len(dataset)

    splits = expanding_window_splits(
        n,
        look_back,
        horizon,
        folds=cfg["validation"]["folds"],
    )

    learning = LearningModule()

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(splits):

        print(f"\nFold {fold+1}")

        train_loader = DataLoader(
            Subset(dataset, list(train_idx)),
            batch_size=trainer_cfg["batch_size"],
            shuffle=True,
        )

        test_loader = DataLoader(
            Subset(dataset, list(test_idx)),
            batch_size=trainer_cfg["batch_size"],
            shuffle=False,
        )

        model = learning.create_forecasting_model(
            input_size=model_cfg["input_size"],
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            horizon=horizon,
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg["bidirectional"],
        )

        trainer = learning.create_trainer(
            model=model,
            learning_rate=trainer_cfg["learning_rate"],
            device=device,
        )

        for epoch in range(trainer_cfg["epochs"]):

            trainer.train_epoch(train_loader)

        val_loss = trainer.validate_epoch(test_loader)

        print(f"Validation loss: {val_loss:.4f}")

        fold_results.append(val_loss)

    print("\nValidation summary")

    print("Mean loss:", np.mean(fold_results))
    print("Std loss:", np.std(fold_results))
    
    # print("\nDetailed fold losses:", fold_results)
    
    # save results
    results = {
        "Validation results": 'MSE loss',
        "mean": float(np.mean(fold_results)),
        "std": float(np.std(fold_results)),
    }
 
    if save_results:

        results_path = os.path.join(exp_dir, "validation_results.json")
        summary_path = os.path.join(exp_dir, "validation_summary.json")

        with open(results_path, "w") as f:
            json.dump(fold_results, f, indent=2)

        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        print("Saved validation results:", results_path)

    return fold_results


def main():

    args = _parse_arguments()

    run_validation(
        dataset_name=args.dataset,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
