import os 
import yaml
import torch

from torch.utils.data import DataLoader, Subset

from ..dataset.insar import create_synthetic_insar_dataset, create_mirmazloumi_2023_dataset
from ..learning import LearningModule
from . import InferenceModule
# from .monitoring_runner import run_monitoring
from subsystems.map.monitoring.runner import run_monitoring
from subsystems.map.evaluation.plots import plot_results
from subsystems.map.registry.model_registry import ModelRegistry


"""
Core ML orchestration. 
"""


# ============= HELPERS =============
def _load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _select_device(device_config):
    if device_config == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_config)


def _build_indices(dataset, split_name, look_back, horizon):
    split = dataset.split_info[split_name]
    indices = []

    for i in range(len(dataset)):
        window_start = i
        window_end = i + look_back + horizon
        if window_start >= split['start_index'] and window_end <= split['end_index']:
            indices.append(i)

    return indices

def _load_experiment_config(cfg):

    exp_dir = os.path.join(
        cfg['experiments']['root_dir'],
        cfg['experiments']['name'],
    )

    exp_config_path = os.path.join(exp_dir, 'config_used.yaml')

    if not os.path.exists(exp_config_path):
        raise RuntimeError(
            f'Experiment config not found: {exp_config_path}'
        )

    with open(exp_config_path, 'r') as f:
        return yaml.safe_load(f)



# ============= MAIN EXPERIMENT =============
def run_lstm_experiment(dataset_name: str, config_path: str):

    # cfg = _load_config(config_path)

    # model_cfg = cfg['model']
    # trainer_cfg = cfg['trainer']
    # dataset_cfg = cfg['dataset']
    # monitor_cfg = cfg['monitoring']
    # infer_cfg = cfg['inference']

    cfg = _load_config(config_path)

    # load experiment config (contains tuned parameters)
    exp_cfg = _load_experiment_config(cfg)

    model_cfg = exp_cfg['model']
    trainer_cfg = exp_cfg['trainer']
    dataset_cfg = exp_cfg['dataset']

    monitor_cfg = cfg['monitoring']
    infer_cfg = cfg['inference']


    device = _select_device(trainer_cfg['device'])

    look_back = trainer_cfg['look_back']
    horizon = trainer_cfg['horizon']



    # ============= DATASET =============
    if dataset_name == 'synthetic':
        dataset = create_synthetic_insar_dataset(
            length=dataset_cfg['length'],
            noise_std=dataset_cfg['noise_std'],
            trend_amplitude=dataset_cfg['trend_amplitude'],
            anomaly_magnitude=dataset_cfg['anomaly_magnitude'],
            look_back=look_back,
            horizon=horizon,
        )
    else:
        dataset = create_mirmazloumi_2023_dataset(
            look_back=look_back,
            horizon=horizon,
        )

    train_loader = DataLoader(
        Subset(dataset, _build_indices(dataset, 'train', look_back, horizon)),
        batch_size=trainer_cfg['batch_size'],
        shuffle=True,
    )

    test_loader = DataLoader(
        Subset(dataset, _build_indices(dataset, 'test', look_back, horizon)),
        batch_size=trainer_cfg['batch_size'],
        shuffle=False,
    )


    # ============= TRAINING =============
    learning = LearningModule()
    
    model = learning.create_forecasting_model(
        input_size=model_cfg['input_size'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        horizon=horizon,
        dropout=model_cfg['dropout'],
        bidirectional=model_cfg['bidirectional'],
    )
    
    # load trained model
    exp_dir = os.path.join(
        cfg['experiments']['root_dir'],
        cfg['experiments']['name']
    )
    
    # redundant? 
    model_path = os.path.join(exp_dir, cfg['experiments']['model_file'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print('Loaded trained model:', model_path)

    trainer = learning.create_trainer(
        model=model,
        learning_rate=trainer_cfg['learning_rate'],
        device=device,
    )

    print('Training model...')
    for epoch in range(trainer_cfg['epochs']):
        train_loss = trainer.train_epoch(train_loader)
        test_loss = trainer.validate_epoch(test_loader)

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | train {train_loss:.4f} | test {test_loss:.4f}')



    # ============= INFERENCE =============
    exp_dir = os.path.join(
        cfg['experiments']['root_dir'],
        cfg['experiments']['name']
    )

    model_path, metadata = ModelRegistry.load_latest_model(exp_dir)

    print('Loading model from registry:')
    print('Version:', metadata['version'])
    print('Model file:', model_path)

    # rebuild architecture
    learning = LearningModule()

    model = learning.create_forecasting_model(
        input_size=model_cfg['input_size'],
        hidden_size=metadata['parameters']['hidden_size'],
        num_layers=metadata['parameters']['num_layers'],
        horizon=horizon,
        dropout=metadata['parameters']['dropout'],
        bidirectional=metadata['parameters']['bidirectional'],
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    inference = InferenceModule()

    predictor = inference.create_predictor(
        model=model,
        device=device,
        look_back=look_back,
        horizon=horizon,
        mc_samples=infer_cfg['mc_samples'],
        sigma_threshold=infer_cfg['sigma_threshold'],
        warmup_factor=monitor_cfg['warmup_factor'],
        calibration_fraction=monitor_cfg['calibration_fraction'],
        persistence=monitor_cfg['persistence'],
        use_model_uncertainty=monitor_cfg['use_model_uncertainty'],
    )

    displacement = dataset.displacement
    time_days = dataset.time_days

    mean_pred, std_pred = predictor.predict_series(displacement)
    residuals = predictor.compute_residuals(displacement, mean_pred)

    monitoring = run_monitoring(residuals, std_pred, predictor, monitor_cfg)

    plot_results(time_days, displacement, mean_pred, std_pred, monitoring)
