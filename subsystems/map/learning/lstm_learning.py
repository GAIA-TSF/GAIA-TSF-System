import os
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .trainer import Trainer

from subsystems.map import dataset
from subsystems.map.registry.model_registry import ModelRegistry 


from ..dataset.insar import (
    create_synthetic_insar_dataset,
    create_mirmazloumi_2023_dataset,
)
# from .lstm_model import LstmModel
# from .trainer import Trainer
from ..learning import LearningModule


# argument parsing 
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

# config loading 
def _load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _select_device(device_config: str) -> torch.device:
    if device_config == 'auto':
        return torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
    return torch.device(device_config)

# index builder (same as pipeline)
def _build_indices(dataset, split_name, look_back, horizon):

    split = dataset.split_info[split_name]
    indices = []

    for i in range(len(dataset)):
        window_start = i
        window_end = i + look_back + horizon

        if window_start >= split['start_index'] and window_end <= split['end_index']:
            indices.append(i)

    return indices



# -------------------------------------------------
# main 
# -------------------------------------------------
def main():
    args = _parse_arguments()
    config = _load_config(args.config)

    # set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42) 

    model_cfg = config['model']
    trainer_cfg = config['trainer']
    dataset_cfg = config['dataset']
    # print(trainer_cfg)

    device = _select_device(trainer_cfg['device'])
    # print(device)
    look_back = trainer_cfg['look_back']
    horizon = trainer_cfg['horizon']

    # -----------------------------
    # Dataset selection
    # -----------------------------
    if args.dataset == 'synthetic':
        dataset = create_synthetic_insar_dataset(
            length=dataset_cfg['length'],
            noise_std=dataset_cfg['noise_std'],
            trend_amplitude=dataset_cfg['trend_amplitude'],
            anomaly_magnitude=dataset_cfg['anomaly_magnitude'],
            look_back=look_back, 
            horizon=horizon
        )
    else:
        dataset = create_mirmazloumi_2023_dataset(
            look_back=look_back,
            horizon=horizon, 
        )
    
    # Train/test split 
    split = dataset.split_info
    train_end = split['train']['end_index']
    test_start = split['test']['start_index']
    test_end = split['test']['end_index']

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
    # Learning module
    # -----------------------------
    learning = LearningModule()

    model = learning.create_forecasting_model(
        input_size=model_cfg['input_size'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        horizon=horizon,
        dropout=model_cfg['dropout'],
        bidirectional=model_cfg['bidirectional'],
    )

    trainer = learning.create_trainer(
        model=model,
        learning_rate=trainer_cfg['learning_rate'],
        device=device,
    )

    # -----------------------------
    # Training
    # -----------------------------
    print('Training starts')

    train_losses = []
    test_losses = []

    exp_dir = os.path.join(
        config['experiments']['root_dir'],
        config['experiments']['name']
    )   

    registry_path = os.path.join(exp_dir, 'model_registry.json') 
    registry = ModelRegistry(registry_path)

    os.makedirs(exp_dir, exist_ok=True)

    config_copy = os.path.join(exp_dir, 'config_used.yaml')

    with open(config_copy, 'w') as f:
        yaml.dump(config, f)

    model_path = os.path.join(
        exp_dir,
        config['experiments']['model_file']
    )

    train_losses, test_losses = trainer.fit(
        train_loader,
        test_loader,
        trainer_cfg['epochs'],
        model_path=model_path
    )

    print('Best model stored in:', model_path) 

    # Register model in registry
    registry_entry = registry.register_model(
        model_file=config["experiments"]["model_file"],
        dataset=args.dataset,
        parameters=config["model"],
        monitoring_cfg=config.get("monitoring", {}),
        metrics={
            "final_train_loss": float(train_losses[-1]),
            "final_test_loss": float(test_losses[-1]),
            "best_test_loss": float(min(test_losses)),
        },
    )

    print('Model registered:')
    print(registry_entry)

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
