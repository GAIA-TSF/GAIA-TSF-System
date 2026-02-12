import argparse
import yaml
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from ..dataset.insar import (
    create_synthetic_insar_dataset,
    create_mirmazloumi_2023_dataset,
)
from .lstm_model import LstmModel
from .trainer import Trainer


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run LSTM learning experiment.',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['synthetic', 'mirmazloumi_2023'],
        help='Dataset type',
    )

    parser.add_argument(
        '--config',
        type=str,
        default='subsystems/map/learning/config.yaml',
        help='Path to config file',
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


def main():
    args = _parse_arguments()
    config = _load_config(args.config)

    model_cfg = config['model']
    trainer_cfg = config['trainer']
    dataset_cfg = config['dataset']
    print(trainer_cfg)

    device = _select_device(trainer_cfg['device'])
    print(device) 

    # -----------------------------
    # Dataset selection
    # -----------------------------
    if args.dataset == 'synthetic':
        dataset = create_synthetic_insar_dataset(
            length=dataset_cfg['length'],
            noise_std=dataset_cfg['noise_std'],
            trend_amplitude=dataset_cfg['trend_amplitude'],
            anomaly_magnitude=dataset_cfg['anomaly_magnitude'],
            look_back=trainer_cfg['look_back'],
            horizon=trainer_cfg['horizon'],
        )
    else:
        dataset = create_mirmazloumi_2023_dataset(
            look_back=trainer_cfg['look_back'],
            horizon=trainer_cfg['horizon'],
        )

    split = dataset.split_info
    train_end = split['train']['end_index']
    test_start = split['test']['start_index']
    test_end = split['test']['end_index']

    look_back = trainer_cfg['look_back']
    horizon = trainer_cfg['horizon']

    train_indices = list(range(0, train_end - look_back - horizon))
    test_indices = list(
        range(
            test_start - look_back,
            test_end - look_back - horizon,
        ),
    )

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
    # Model
    # -----------------------------
    model = LstmModel(
        input_size=model_cfg['input_size'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        output_size=model_cfg['output_size'],
        horizon=model_cfg['horizon'],
        mode=model_cfg['mode'],
        dropout=model_cfg['dropout'],
        bidirectional=model_cfg['bidirectional'],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_cfg['learning_rate'],
        weight_decay=trainer_cfg['weight_decay'],
    )

    loss_fn = torch.nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    # -----------------------------
    # Training
    # -----------------------------
    print('Training starts') 
    train_losses = []
    test_losses = []

    for epoch in range(trainer_cfg['epochs']):
        train_loss = trainer.train_epoch(train_loader)
        test_loss = trainer.validate_epoch(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch == 0 or epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d} | '
                f'Train: {train_loss:.4f} | '
                f'Test: {test_loss:.4f}',
            )

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'LSTM Learning – {args.dataset}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

