import torch
import torch.nn as nn
import torch.optim as optim

from .lstm_model import LstmModel
from .trainer import Trainer


class LearningModule:
    """Learning module for MAP.

    Provides LSTM-based learning for multiple anomaly-detection
    paradigms.
    """

    @staticmethod
    def create_forecasting_model(
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> LstmModel:
        return LstmModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=horizon,
            mode='forecasting',
            horizon=horizon,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    @staticmethod 
    def create_reconstruction_model(
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> LstmModel:
        return LstmModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,
            mode='reconstruction',
            horizon=1,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    @staticmethod
    def create_trainer(
        model: torch.nn.Module,
        learning_rate: float,
        device: torch.device,
    ) -> Trainer:
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        loss_fn = nn.MSELoss()

        return Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
