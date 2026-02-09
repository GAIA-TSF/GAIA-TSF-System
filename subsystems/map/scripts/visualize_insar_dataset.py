import argparse

import matplotlib.pyplot as plt

from ..dataset.insar import (
    create_mirmazloumi_2023_dataset,
    create_synthetic_insar_dataset,
) 


"""
Usage: 

python3 -m subsystems.map.scripts.visualize_insar_dataset --dataset mirmazloumi_2023 

python3 -m subsystems.map.scripts.visualize_insar_dataset --dataset synthetic

"""

def plot_insar_dataset(dataset, title: str):
    plt.figure(figsize=(10, 4))

    color_map = {
        'train': 'blue',
        'test': 'green',
        'anomaly': 'red',
        'full': 'black',
    }

    for split_key, split in dataset.split_info.items():
        start = split['start_index']
        end = split['end_index']

        plt.plot(
            dataset.time_days[start:end],
            dataset.displacement[start:end],
            color=color_map.get(split_key, 'black'),
            marker='o',
            linewidth=1.5,
            label=split['label'],
        )

    plt.xlabel('Time [days]')
    plt.ylabel('LOS displacement [mm]')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _parse_arguments():
    parser = argparse.ArgumentParser(
        description='Visualize InSAR datasets used in MAP.',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=[
            'mirmazloumi_2023',
            'synthetic',
        ],
        help='Dataset to visualize',
    )

    parser.add_argument(
        '--anomaly-magnitude',
        type=float,
        default=20.0,
        help=(
            'Total anomaly displacement [mm] applied during the anomaly '
            'period (synthetic dataset only)'
        ),
    )

    return parser.parse_args()


def _load_dataset(
    dataset_name: str,
    anomaly_magnitude: float,
):
    if dataset_name == 'mirmazloumi_2023':
        return (
            create_mirmazloumi_2023_dataset(
                look_back=12,
                horizon=5,
            ),
            'Mirmazloumi et al. (2023) – InSAR Time Series',
        )

    if dataset_name == 'synthetic':
        return (
            create_synthetic_insar_dataset(
                length=80,
                noise_std=0.5,
                trend_amplitude=20.0,
                anomaly_magnitude=anomaly_magnitude,
                look_back=12,
                horizon=5,
            ),
            (
                'Synthetic InSAR Time Series '
                f'(Anomaly = {anomaly_magnitude:.1f} mm)'
            ),
        )

    raise ValueError(f'Unknown dataset: {dataset_name}')



if __name__ == '__main__':
    args = _parse_arguments()

    dataset, title = _load_dataset(
        dataset_name=args.dataset,
        anomaly_magnitude=args.anomaly_magnitude,
    )

    plot_insar_dataset(
        dataset=dataset,
        title=title,
    )

