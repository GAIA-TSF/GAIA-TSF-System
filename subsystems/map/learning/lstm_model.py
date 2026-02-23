import torch
import torch.nn as nn

"""
This is the prototype LSTM module for InSAR time series modelling.
"""


class LstmModel(nn.Module):
    """Generic LSTM model for time-series learning.

    Supports:
    - forecasting (predict future steps)
    - reconstruction (autoencoder-style)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        horizon: int,
        mode: str,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        if mode not in {'forecasting', 'reconstruction'}:
            raise ValueError(
                'mode must be forecasting or reconstruction',
            )

        self._mode = mode
        self._horizon = horizon
        self._bidirectional = bidirectional

        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # TODO: set dropout 
            # dropout=dropout if num_layers > 1 else 0.2,
            batch_first=True,
            bidirectional=bidirectional,
        )

        direction_multiplier = 2 if bidirectional else 1

        self._head = nn.Linear(
            hidden_size * direction_multiplier,
            output_size if mode == 'reconstruction' else horizon,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, time, features)
        """
        outputs, _ = self._lstm(inputs)

        if self._mode == 'forecasting':
            last_hidden = outputs[:, -1, :]
            return self._head(last_hidden)

        # reconstruction
        return self._head(outputs)
