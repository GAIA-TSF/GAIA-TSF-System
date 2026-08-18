"""Causal LSTM predictive-model plugin for temporal MAP features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from subsystems.map.core.interfaces import PredictionResult, PredictiveModel
from subsystems.map.core.registry import register_model


LOGGER = logging.getLogger(__name__)


@register_model('lstm')
class LSTMModel(PredictiveModel):
    """LSTM forecaster over causal per-pixel sequences of DAG features.

    The plugin learns normal one-step-ahead deformation behaviour. Temporal
    windows are assembled by :class:`DatasetBuilder`; this model neither
    derives nor modifies engineered features.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize configuration and import the optional PyTorch backend."""
        super().__init__(config)
        try:
            import torch
            import torch.nn as nn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'LSTMModel requires the optional PyTorch dependency.',
            ) from exc
        self._torch = torch
        self._nn = nn
        self.look_back = self._positive_integer('look_back')
        self.horizon = self._positive_integer('horizon')
        if self.horizon != 1:
            raise ValueError('LSTMModel currently supports horizon: 1 only.')
        self.hidden_size = self._positive_integer('hidden_size')
        self.num_layers = self._positive_integer('num_layers')
        self.dropout = self._unit_interval('dropout')
        self.bidirectional = bool(self.config.get('bidirectional', False))
        if self.bidirectional:
            raise ValueError('Bidirectional LSTM is not causal and is not supported.')
        training = self._mapping('training')
        optimizer = self._mapping('optimizer')
        self.batch_size = self._positive_integer_from(training, 'batch_size')
        self.epochs = self._positive_integer_from(training, 'epochs')
        self.learning_rate = self._positive_float_from(optimizer, 'lr')
        self.device = self._device(str(self.config.get('device', 'cpu')))
        self._network: Any | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._target_mean: float | None = None
        self._target_std: float | None = None
        self.training_history: list[float] = []
        self.validation_history: list[float] = []
        self._validation_data: tuple[np.ndarray, np.ndarray] | None = None

    def sequence_spec(self) -> tuple[int, int]:
        """Return the causal temporal-window specification for this model."""
        return self.look_back, self.horizon

    def set_validation_data(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Store held-out causal sequences for epoch-level validation loss."""
        sequences = self._validate_sequences(features)
        observed = np.asarray(targets, dtype=np.float64)
        if observed.ndim != 1 or observed.size != sequences.shape[0]:
            raise ValueError('LSTM validation targets must match input sequences.')
        if not np.all(np.isfinite(observed)):
            raise ValueError('LSTM validation targets must be finite.')
        self._validation_data = (sequences, observed)

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Fit the LSTM using normalized causal sequences.

        Args:
            features: Array shaped ``(samples, look_back, features)``.
            targets: One observed target per sequence.
        """
        sequences, observed = self._validate_training_data(features, targets)
        self._seed_torch()
        self._feature_mean = np.mean(sequences, axis=(0, 1))
        self._feature_std = self._safe_std(np.std(sequences, axis=(0, 1)))
        self._target_mean = float(np.mean(observed))
        self._target_std = float(self._safe_std(np.asarray(np.std(observed))))
        normalized_features = (sequences - self._feature_mean) / self._feature_std
        normalized_targets = (observed - self._target_mean) / self._target_std
        self._network = self._build_network(normalized_features.shape[-1]).to(self.device)
        optimizer = self._torch.optim.Adam(
            self._network.parameters(),
            lr=self.learning_rate,
        )
        criterion = self._nn.MSELoss()
        feature_tensor = self._torch.as_tensor(normalized_features, dtype=self._torch.float32)
        target_tensor = self._torch.as_tensor(normalized_targets, dtype=self._torch.float32)
        validation_tensors = self._normalized_validation_tensors()
        generator = self._torch.Generator(device='cpu')
        generator.manual_seed(self._seed_value())
        self.training_history = []
        self.validation_history = []
        for epoch in range(self.epochs):
            self._network.train()
            total_loss = 0.0
            total_count = 0
            order = self._torch.randperm(feature_tensor.shape[0], generator=generator)
            for start in range(0, feature_tensor.shape[0], self.batch_size):
                batch_index = order[start:start + self.batch_size]
                inputs = feature_tensor[batch_index].to(self.device)
                batch_targets = target_tensor[batch_index].to(self.device)
                optimizer.zero_grad()
                output = self._network(inputs).squeeze(-1)
                loss = criterion(output, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * inputs.shape[0]
                total_count += int(inputs.shape[0])
            epoch_loss = total_loss / total_count
            self.training_history.append(epoch_loss)
            validation_loss = self._validation_loss(criterion, validation_tensors)
            if validation_loss is not None:
                self.validation_history.append(validation_loss)
                LOGGER.info(
                    'LSTM epoch %d/%d training_loss=%.6f validation_loss=%.6f',
                    epoch + 1,
                    self.epochs,
                    epoch_loss,
                    validation_loss,
                )
            else:
                LOGGER.info(
                    'LSTM epoch %d/%d training_loss=%.6f',
                    epoch + 1,
                    self.epochs,
                    epoch_loss,
                )

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Predict target deformation for causal input sequences."""
        if self._network is None or self._feature_mean is None or self._feature_std is None:
            raise RuntimeError('LSTMModel must be trained or loaded before prediction.')
        if self._target_mean is None or self._target_std is None:
            raise RuntimeError('LSTM target normalization is unavailable.')
        sequences = self._validate_sequences(features)
        if sequences.shape[-1] != self._feature_mean.size:
            raise ValueError('LSTM input feature count differs from the trained model.')
        normalized = (sequences - self._feature_mean) / self._feature_std
        self._network.eval()
        with self._torch.no_grad():
            output = self._network(
                self._torch.as_tensor(normalized, dtype=self._torch.float32).to(self.device),
            ).squeeze(-1)
        predictions = output.detach().cpu().numpy()
        return PredictionResult(predictions * self._target_std + self._target_mean)

    def save(self, path: Path) -> None:
        """Persist architecture, normalization and learned network parameters."""
        if self._network is None:
            raise RuntimeError('Cannot save an untrained LSTMModel.')
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                'config': self.config,
                'feature_mean': self._feature_mean,
                'feature_std': self._feature_std,
                'target_mean': self._target_mean,
                'target_std': self._target_std,
                'input_size': int(self._feature_mean.size),
                'state_dict': self._network.state_dict(),
                'training_history': self.training_history,
                'validation_history': self.validation_history,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> 'LSTMModel':
        """Load a persisted LSTM model on the configured inference device."""
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'LSTMModel requires the optional PyTorch dependency.',
            ) from exc
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or 'config' not in payload:
            raise TypeError(f'Invalid LSTM model artifact: {path}')
        model = cls(payload['config'])
        input_size = int(payload['input_size'])
        model._network = model._build_network(input_size).to(model.device)
        model._network.load_state_dict(payload['state_dict'])
        model._feature_mean = np.asarray(payload['feature_mean'], dtype=np.float64)
        model._feature_std = np.asarray(payload['feature_std'], dtype=np.float64)
        model._target_mean = float(payload['target_mean'])
        model._target_std = float(payload['target_std'])
        model.training_history = [float(value) for value in payload.get('training_history', [])]
        model.validation_history = [
            float(value) for value in payload.get('validation_history', [])
        ]
        return model

    def _normalized_validation_tensors(self) -> tuple[Any, Any] | None:
        """Normalize stored validation data with training-only statistics."""
        if self._validation_data is None:
            return None
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError('LSTM feature normalization is unavailable.')
        if self._target_mean is None or self._target_std is None:
            raise RuntimeError('LSTM target normalization is unavailable.')
        features, targets = self._validation_data
        normalized_features = (features - self._feature_mean) / self._feature_std
        normalized_targets = (targets - self._target_mean) / self._target_std
        return (
            self._torch.as_tensor(normalized_features, dtype=self._torch.float32).to(
                self.device,
            ),
            self._torch.as_tensor(normalized_targets, dtype=self._torch.float32).to(
                self.device,
            ),
        )

    def _validation_loss(
        self,
        criterion: Any,
        validation_tensors: tuple[Any, Any] | None,
    ) -> float | None:
        """Evaluate held-out normalized loss without updating model weights."""
        if validation_tensors is None:
            return None
        features, targets = validation_tensors
        self._network.eval()
        with self._torch.no_grad():
            loss = criterion(self._network(features).squeeze(-1), targets)
        return float(loss.detach().cpu())

    def _build_network(self, input_size: int) -> Any:
        """Create the causal forecasting network from this plugin's settings."""
        nn = self._nn
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        dropout = self.dropout

        class Network(nn.Module):
            """One-step LSTM forecaster with a linear output head."""

            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, inputs: Any) -> Any:
                """Forecast one target from the final causal hidden state."""
                outputs, _ = self.lstm(inputs)
                return self.head(self.dropout(outputs[:, -1, :]))

        return Network()

    def _validate_training_data(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate finite, aligned training sequences and targets."""
        sequences = self._validate_sequences(features)
        observed = np.asarray(targets, dtype=np.float64)
        if observed.ndim != 1 or observed.size != sequences.shape[0]:
            raise ValueError('LSTM targets must be one value per input sequence.')
        if not np.all(np.isfinite(observed)):
            raise ValueError('LSTM targets must be finite.')
        return sequences, observed

    def _validate_sequences(self, features: np.ndarray) -> np.ndarray:
        """Validate finite three-dimensional causal model input."""
        sequences = np.asarray(features, dtype=np.float64)
        if sequences.ndim != 3:
            raise ValueError(
                'LSTM features must have shape (samples, look_back, features).',
            )
        if sequences.shape[0] == 0 or sequences.shape[1] != self.look_back:
            raise ValueError('LSTM input has an invalid sample count or look_back.')
        if not np.all(np.isfinite(sequences)):
            raise ValueError('LSTM features must be finite.')
        return sequences

    def _seed_torch(self) -> None:
        """Seed PyTorch and request deterministic operators where available."""
        seed = self._seed_value()
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)
        self._torch.use_deterministic_algorithms(True, warn_only=True)

    def _seed_value(self) -> int:
        """Return the pipeline seed, falling back to the plugin configuration."""
        return int(getattr(self, '_random_seed', self.config.get('random_seed', 42)))

    def _device(self, configured: str) -> Any:
        """Validate and return a configured torch device."""
        device = self._torch.device(configured)
        if device.type == 'cuda' and not self._torch.cuda.is_available():
            raise ValueError('LSTM device cuda was configured but is unavailable.')
        return device

    def _mapping(self, name: str) -> dict[str, Any]:
        """Return a required model configuration mapping."""
        value = self.config.get(name)
        if not isinstance(value, dict):
            raise ValueError(f'LSTM configuration {name} must be a mapping.')
        return value

    def _positive_integer(self, name: str) -> int:
        """Read a positive integer from the top-level model configuration."""
        return self._positive_integer_from(self.config, name)

    @staticmethod
    def _positive_integer_from(config: dict[str, Any], name: str) -> int:
        """Read a positive integer from a configuration mapping."""
        value = config.get(name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'LSTM configuration {name} must be a positive integer.')
        return value

    @staticmethod
    def _positive_float_from(config: dict[str, Any], name: str) -> float:
        """Read a positive floating point value from configuration."""
        value = float(config.get(name, 0.0))
        if value <= 0:
            raise ValueError(f'LSTM configuration {name} must be positive.')
        return value

    def _unit_interval(self, name: str) -> float:
        """Read a value in the closed unit interval from model configuration."""
        value = float(self.config.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'LSTM configuration {name} must be in [0, 1].')
        return value

    @staticmethod
    def _safe_std(values: np.ndarray) -> np.ndarray:
        """Replace zero standard deviations with one for stable normalization."""
        return np.where(values > np.finfo(np.float64).eps, values, 1.0)
