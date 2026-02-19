import numpy as np
import torch

from ..dataset import DatasetModule

"""
This class defines InSAR dataset of displacement for PyTorch 
X: inputs
y: targets 
- create_mirmazloumi_2023_dataset()
- create_synthetic_insar_datas() 
"""


class InSARDataset(DatasetModule):
    """InSAR time-series dataset with explicit time axis and split info."""

    def __init__(
        self,
        displacement: np.ndarray,
        time_days: np.ndarray,
        split_info: dict,
        look_back: int,
        horizon: int,
    ):
        self._displacement = displacement.astype(np.float32)
        self._time_days = time_days.astype(np.int32)
        self._split_info = split_info
        self._look_back = look_back
        self._horizon = horizon

        self._inputs = None
        self._targets = None

        self._validate_inputs()
        self.build()

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int):
        return self._inputs[index], self._targets[index]

    def build(self):
        inputs = []
        targets = []

        series_length = len(self._displacement)

        for idx in range(
            0,
            series_length - self._look_back - self._horizon,
        ):
            inputs.append(
                self._displacement[idx : idx + self._look_back],
            )
            targets.append(
                self._displacement[
                    idx + self._look_back : idx + self._look_back + self._horizon
                ],
            )

        inputs_array = np.array(
            inputs,
            dtype=np.float32,
        )
        targets_array = np.array(
            targets,
            dtype=np.float32,
        )

        self._inputs = torch.from_numpy(
            inputs_array,
        ).unsqueeze(-1)

        self._targets = torch.from_numpy(
            targets_array,
        )

    def _validate_inputs(self):
        if self._displacement.ndim != 1:
            raise ValueError('displacement must be 1D')

        if self._time_days.ndim != 1:
            raise ValueError('time_days must be 1D')

        if len(self._displacement) != len(self._time_days):
            raise ValueError('time_days and displacement must align')

    @property
    def time_days(self) -> np.ndarray:
        return self._time_days

    @property
    def displacement(self) -> np.ndarray:
        return self._displacement

    @property
    def split_info(self) -> dict:
        return self._split_info


def create_mirmazloumi_2023_dataset(
    look_back: int,
    horizon: int,
) -> InSARDataset:
    """Reproduce Fig. 4 time series from Mirmazloumi et al. (2023)."""

    train_displacement = np.array(
        [
            0.0,
            4.2,
            5.1,
            1.8,
            0.5,
            4.8,
            5.8,
            -1.5,
            3.1,
            -0.2,
            -3.5,
            1.2,
            2.1,
            -5.8,
            -0.5,
            -0.2,
            -3.8,
            3.5,
            5.5,
            2.8,
            4.9,
            2.8,
            13.1,
            4.1,
            4.5,
            6.7,
            4.8,
            1.9,
            0.2,
            7.1,
            7.3,
            2.2,
            -1.1,
            2.1,
            -6.8,
            4.1,
            2.9,
            13.5,
            2.1,
            6.2,
            -4.5,
            -3.1,
            -0.2,
            -0.5,
            -3.5,
            2.1,
            -5.2,
            2.2,
            -1.8,
        ]
    )

    test_displacement = np.array(
        [
            -2.5,
            -13.5,
            -4.5,
            2.5,
            -0.5,
            1.1,
            -3.1,
            2.2,
            -5.5,
            -6.5,
            -4.1,
            -11.8,
            0.5,
            9.2,
            0.8,
        ]
    )

    anomaly_displacement = np.array(
        [
            1.5,
            6.5,
            0.1,
            1.8,
            0.6,
        ]
    )

    displacement = np.concatenate(
        [
            train_displacement,
            test_displacement,
            anomaly_displacement,
        ],
    )

    time_days = np.arange(len(displacement)) * 5

    split_info = {
        'train': {
            'start_index': 0,
            'end_index': len(train_displacement),
            'label': 'Train',
        },
        'test': {
            'start_index': len(train_displacement),
            'end_index': len(train_displacement) + len(test_displacement),
            'label': 'Test',
        },
        'anomaly': {
            'start_index': (len(train_displacement) + len(test_displacement)),
            'end_index': len(displacement),
            'label': 'Anomaly Period',
        },
    }

    return InSARDataset(
        displacement=displacement,
        time_days=time_days,
        split_info=split_info,
        look_back=look_back,
        horizon=horizon,
    )


def create_synthetic_insar_dataset(
    length: int,
    noise_std: float,
    trend_amplitude: float,
    anomaly_magnitude: float,
    look_back: int,
    horizon: int,
    seed: int = 42,
) -> InSARDataset:
    """Generate synthetic InSAR dataset with train/test/anomaly split.

    Parameters
    ----------
    length : int
        Total number of time steps.
    noise_std : float
        Standard deviation of Gaussian noise.
    trend_amplitude : float
        Total linear trend amplitude (mm).
    anomaly_magnitude : float
        Total displacement added during anomaly period (mm).
    look_back : int
        Look-back window length.
    horizon : int
        Forecast horizon.
    seed : int
        Random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)

    # --- Base signal ---
    noise = rng.normal(0.0, noise_std, length)
    trend = np.linspace(0.0, -trend_amplitude, length)
    displacement = np.cumsum(noise) + trend

    # --- Split indices ---
    train_end = int(0.7 * length)
    test_end = int(0.9 * length)

    anomaly_start = test_end
    anomaly_end = length

    # --- Inject anomaly ---
    anomaly_length = anomaly_end - anomaly_start

    anomaly_signal = np.linspace(
        0.0,
        anomaly_magnitude,
        anomaly_length,
    )

    displacement[anomaly_start:anomaly_end] += anomaly_signal

    # --- Time axis (5-day sampling) ---
    time_days = np.arange(length) * 5

    split_info = {
        'train': {
            'start_index': 0,
            'end_index': train_end,
            'label': 'Train',
        },
        'test': {
            'start_index': train_end,
            'end_index': test_end,
            'label': 'Test',
        },
        'anomaly': {
            'start_index': anomaly_start,
            'end_index': anomaly_end,
            'label': 'Anomaly Period',
        },
    }

    return InSARDataset(
        displacement=displacement,
        time_days=time_days,
        split_info=split_info,
        look_back=look_back,
        horizon=horizon,
    )
