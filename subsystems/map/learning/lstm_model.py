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
        mode: str,
    ):
        super().__init__()

        if mode not in {'forecasting', 'reconstruction'}:
            raise ValueError(
                'mode must be forecasting or reconstruction',
            )

        self._mode = mode

        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self._head = nn.Linear(
            hidden_size,
            output_size,
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
