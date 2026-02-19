import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from ..dataset.insar import create_mirmazloumi_2023_dataset
from ..learning import LearningModule

"""
Usage: 
python3 -m subsystems.map.scripts.train_lstm_mirmazloumi

"""


def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    print('Configuration')
    look_back = 12
    horizon = 5
    batch_size = 8
    epochs = 500
    learning_rate = 1e-3

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu',
    )

    # -------------------------------
    # Dataset
    # -------------------------------
    dataset = create_mirmazloumi_2023_dataset(
        look_back=look_back,
        horizon=horizon,
    )

    split = dataset.split_info

    train_end = split['train']['end_index']
    test_start = split['test']['start_index']
    test_end = split['test']['end_index']

    # Dataset indices correspond to sliding windows
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

    def plot_learning_curves(
        train_losses: list[float],
        test_losses: list[float],
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
        plt.title('LSTM Learning Curves – Mirmazloumi (2023)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # -------------------------------
    # Training loop
    # -------------------------------
    train_losses = []
    test_losses = []

    print('Starting LSTM training on Mirmazloumi (2023) dataset')

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(
            train_loader,
        )

        test_loss = trainer.validate_epoch(
            test_loader,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0 or epoch == 1:
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
    )


if __name__ == '__main__':
    main()
