import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch
from subsystems.map.learning.lstm_model import LstmModel
from subsystems.map.learning.trainer import Trainer


class TestLSTMTraining:
    """
    Tests for LSTM training functionality.
    """

    def test_model_forward_pass(self):
        """
        Verify the LSTM model produces outputs with correct shape.
        """

        model = LstmModel(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            output_size=1,
            horizon=3,
            mode='forecasting',
        )

        x = torch.randn(4, 5, 1)  # batch, look_back, features

        y = model(x)

        assert y.shape == (4, 3)

    def test_trainer_single_epoch(self):
        """
        Verify trainer runs one training epoch without error.
        """

        model = LstmModel(
            input_size=1,
            hidden_size=8,
            num_layers=1,
            output_size=1,
            horizon=2,
            mode='forecasting',
        )

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.MSELoss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device('cpu'),
        )

        inputs = torch.randn(10, 5, 1)
        targets = torch.randn(10, 2)

        dataset = list(zip(inputs, targets))

        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=2)

        loss = trainer.train_epoch(loader)

        assert loss >= 0
