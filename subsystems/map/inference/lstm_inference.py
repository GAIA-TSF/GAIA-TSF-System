
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..dataset.insar import (
    create_synthetic_insar_dataset,
    create_mirmazloumi_2023_dataset,
)
from ..learning import LearningModule
from . import InferenceModule

"""
Run prediction on synthetic dataset. 
- train model quickly
- run prediction
- plot observed vs predicted vs anomaly score.

Usage: 
python3 -m subsystems.map.inference.lstm_inference --dataset synthetic

python3 -m subsystems.map.inference.lstm_inference --dataset mirmazloumi_2023

"""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run LSTM inference experiment.',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['synthetic', 'mirmazloumi_2023'],
    )

    parser.add_argument(
        '--config',
        type=str,
        default='subsystems/map/learning/config.yaml',
    )

    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _select_device(device_config: str) -> torch.device:
    if device_config == 'auto':
        return torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
    return torch.device(device_config)


# ---------------------------------------------------------------------
# Window-safe split builder (important!)
# ---------------------------------------------------------------------
def _build_indices(dataset, split_name, look_back, horizon):
    split = dataset.split_info[split_name]

    indices = []
    for i in range(len(dataset)):
        window_start = i
        window_end = i + look_back + horizon

        if (
            window_start >= split['start_index']
            and window_end <= split['end_index']
        ):
            indices.append(i)

    return indices


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = _parse_arguments()
    config = _load_config(args.config)

    model_cfg = config['model']
    trainer_cfg = config['trainer']
    dataset_cfg = config['dataset']

    device = _select_device(trainer_cfg['device'])

    look_back = trainer_cfg['look_back']
    horizon = trainer_cfg['horizon']

    # -----------------------------
    # Dataset
    # -----------------------------
    if args.dataset == 'synthetic':
        dataset = create_synthetic_insar_dataset(
            length=dataset_cfg['length'],
            noise_std=dataset_cfg['noise_std'],
            trend_amplitude=dataset_cfg['trend_amplitude'],
            anomaly_magnitude=dataset_cfg['anomaly_magnitude'],
            look_back=look_back,
            horizon=horizon,
        )
    else:
        dataset = create_mirmazloumi_2023_dataset(
            look_back=look_back,
            horizon=horizon,
        )

    # -----------------------------
    # Build dataloaders
    # -----------------------------
    train_indices = _build_indices(dataset, 'train', look_back, horizon)
    test_indices = _build_indices(dataset, 'test', look_back, horizon)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=trainer_cfg['batch_size'],
        shuffle=True,
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=trainer_cfg['batch_size'],
        shuffle=False,
    )

    # -----------------------------
    # Model + training
    # -----------------------------
    learning = LearningModule()

    model = learning.create_forecasting_model(
        input_size=model_cfg['input_size'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        horizon=horizon,
        dropout=model_cfg['dropout'],
        bidirectional=model_cfg['bidirectional'],
    )

    trainer = learning.create_trainer(
        model=model,
        learning_rate=trainer_cfg['learning_rate'],
        device=device,
    )

    print('Training model before inference...')
    for epoch in range(trainer_cfg['epochs']):
        train_loss = trainer.train_epoch(train_loader)
        test_loss = trainer.validate_epoch(test_loader)

        if epoch == 0 or epoch % 20 == 0:
            if np.isnan(test_loss):
                print(f'Epoch {epoch:03d} | train {train_loss:.4f} | no validation windows')
            else:
                print(f'Epoch {epoch:03d} | train {train_loss:.4f} | test {test_loss:.4f}')


    # -----------------------------
    # Inference
    # -----------------------------
    inference = InferenceModule()

    inf_cfg = config['inference']

    predictor = inference.create_predictor(
        model=model,
        device=device,
        look_back=look_back,
        horizon=horizon,
        mc_samples=inf_cfg['mc_samples'],
        sigma_threshold=inf_cfg['sigma_threshold'],
    )

    displacement = dataset.displacement
    time_days = dataset.time_days

    # old prediction 
    # prediction = predictor.predict_series(displacement)
    # residuals = predictor.compute_residuals(displacement, prediction)
    # score = predictor.anomaly_score(residuals)

    # Uncertainty added
    print(vars(predictor))
     
    mean_pred, std_pred = predictor.predict_series(displacement)
    residuals = predictor.compute_residuals(displacement, mean_pred)
    D, threshold, anomaly_mask = predictor.detect_anomaly(residuals, std_pred)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 6))

    # --- Prediction plot ---
    plt.subplot(2, 1, 1)

    plt.plot(time_days, displacement, '.', color='black', label='Observed')
    plt.plot(time_days, mean_pred, '.', color='blue', label='Predicted')

    # uncertainty band
    upper = mean_pred + threshold
    lower = mean_pred - threshold
    plt.fill_between(time_days, lower, upper, color='blue', alpha=0.2, label='Uncertainty')

    # anomalies
    plt.scatter(
        time_days[anomaly_mask],
        displacement[anomaly_mask],
        color='red',
        s=40,
        label='Detected anomaly',
    )

    plt.legend()
    plt.title('Prediction with uncertainty bounds')
    plt.xlabel('Time (days)')
    plt.ylabel('Displacement')

    # --- Magnitude D plot ---
    plt.subplot(2, 1, 2)

    plt.plot(time_days, D, color='red', label='Anomaly magnitude D')
    plt.plot(time_days, threshold, color='black', linestyle='--', label='Threshold')
    plt.legend()
    plt.title('Anomaly score')
    plt.xlabel('Time (days)')
    plt.ylabel('|Residual|')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
