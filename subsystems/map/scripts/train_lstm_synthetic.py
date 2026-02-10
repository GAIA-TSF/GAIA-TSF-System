
import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from ..dataset.insar import create_synthetic_insar_dataset
from ..learning import LearningModule

"""
Usage: 
python3 -m subsystems.map.scripts.train_lstm_synthetic \
    --anomaly-magnitude 5.0
"""

def _parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train LSTM on synthetic InSAR dataset.',
    )

    parser.add_argument(
        '--anomaly-magnitude',
        type=float,
        default=20.0,
        help='Total anomaly displacement [mm] (synthetic only)',
    )

    return parser.parse_args()


def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    look_back = 12
    horizon = 5
    batch_size = 8
    epochs = 200
    learning_rate = 1e-3

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu',
    )

    args = _parse_arguments()

    # -------------------------------
    # Dataset
    # -------------------------------
    dataset = create_synthetic_insar_dataset(
        length=80,
        noise_std=0.5,
        trend_amplitude=20.0,
        anomaly_magnitude=args.anomaly_magnitude,
        look_back=look_back,
        horizon=horizon,
    )

    split = dataset.split_info

    train_end = split['train']['end_index']
    test_start = split['test']['start_index']
    test_end = split['test']['end_index']

    train_indices = list(
        range(
            0,
            train_end - look_back - horizon,
        ),
    )

    test_indices = list(
        range(
            test_start - look_back,
            test_end - look_back - horizon,
        ),
    )

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    test_dataset = Subset(
        dataset,
        test_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # -------------------------------
    # Model & trainer
    # -------------------------------
    learning = LearningModule()

    model = learning.create_forecasting_model(
        input_size=1,
        hidden_size=64,
        num_layers=1,
        horizon=horizon,
    )

    trainer = learning.create_trainer(
        model=model,
        learning_rate=learning_rate,
        device=device,
    )

    # -------------------------------
    # Training loop
    # -------------------------------
    train_losses = []
    test_losses = []

    print(
        'Starting LSTM training on synthetic InSAR dataset '
        f'(anomaly magnitude = {args.anomaly_magnitude:.1f} mm)',
    )

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(
            train_loader,
        )

        test_loss = trainer.validate_epoch(
            test_loader,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d} | '
                f'Train loss: {train_loss:.4f} | '
                f'Test loss: {test_loss:.4f}',
            )

    print('Training completed')

    # -------------------------------
    # Plot learning curves
    # -------------------------------
    plot_learning_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        anomaly_magnitude=args.anomaly_magnitude,
    )


def plot_learning_curves(
    train_losses: list[float],
    test_losses: list[float],
    anomaly_magnitude: float,
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 4))

    plt.plot(
        epochs,
        train_losses,
        label='Train loss',
        color='blue',
        linewidth=2,
    )

    plt.plot(
        epochs,
        test_losses,
        label='Test loss',
        color='green',
        linewidth=2,
    )

    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.title(
        'LSTM Learning Curves – Synthetic InSAR '
        f'(Anomaly = {anomaly_magnitude:.1f} mm)',
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
