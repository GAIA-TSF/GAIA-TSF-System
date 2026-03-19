import os 
import argparse
import yaml 
import numpy as np 
import torch 

# import matplotlib.pyplot as plt
from subsystems.map.utils.utils import _load_config, _select_device
from subsystems.map.utils.builders import create_dataloaders, create_model
from subsystems.map.monitoring.runner import run_monitoring 
from ..dataset.insar import create_synthetic_insar_dataset, create_mirmazloumi_2023_dataset
from ..learning import LearningModule
from ..inference import InferenceModule
from ..evaluation.plots import plot_results

""" 
Entry point only. 
"""

def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run LSTM monitoring experiment"
    )

    parser.add_argument(
        '--dataset',
        required=True,
        choices=['synthetic', 'mirmazloumi_2023'],
    )

    parser.add_argument(
        '--config',
        default='subsystems/map/learning/config.yaml',
    )

    return parser.parse_args()


# ============= Window-safe split builder (important!) =============
def _build_indices(dataset, split_name, look_back, horizon):
    split = dataset.split_info[split_name]

    indices = []
    for i in range(len(dataset)):
        window_start = i
        window_end = i + look_back + horizon

        if window_start >= split['start_index'] and window_end <= split['end_index']:
            indices.append(i)

    return indices


# ============= Main =============
def main():
    args = _parse_arguments()
    config = _load_config(args.config)

    model_cfg = config['model']
    trainer_cfg = config['trainer']
    dataset_cfg = config['dataset']
    inference_cfg = config['inference']
    monitoring_cfg = config['monitoring']

    device = _select_device(trainer_cfg['device'])

    look_back = trainer_cfg['look_back']
    horizon = trainer_cfg['horizon']

    mc_samples = inference_cfg['mc_samples']
    sigma_threshold = inference_cfg['sigma_threshold']
    warmup_factor = monitoring_cfg['warmup_factor']
    calibration_fraction = monitoring_cfg['calibration_fraction']
    persistence = monitoring_cfg['persistence'] 
    use_model_uncertainty = monitoring_cfg['use_model_uncertainty'] 
    
    # ============= Dataset =============
    if args.dataset == 'synthetic':
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

    
    # ============= Build dataloaders =============
    train_indices = _build_indices(dataset, 'train', look_back, horizon)
    test_indices = _build_indices(dataset, 'test', look_back, horizon)

    train_loader, test_loader = create_dataloaders(
        dataset,
        train_indices,
        test_indices,
        trainer_cfg['batch_size']
    )


    # ============= Model =============
    learning = LearningModule()
    model = create_model(learning, model_cfg, horizon) 

    # ============= load trained weights =============
    exp_dir = config['experiments']['root_dir']
    model_path = os.path.join(
        exp_dir,
        config['experiments']['name'], 
        config['experiments']['model_file']
    )
    print(f'Loading model from: {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    
    # ============= Inference =============    
    predictor = InferenceModule.create_predictor(
        model=model,
        device=device,
        look_back=look_back,
        horizon=horizon,
        mc_samples=mc_samples,
        sigma_threshold=sigma_threshold,
        warmup_factor=warmup_factor,
        calibration_fraction=calibration_fraction,
        persistence=persistence,
        use_model_uncertainty=use_model_uncertainty
    )

    displacement = dataset.displacement
    time_days = dataset.time_days

    mean_pred, std_pred = predictor.predict_series(displacement)
    residuals = predictor.compute_residuals(displacement, mean_pred)

    mon = run_monitoring(residuals, std_pred, predictor, monitoring_cfg)

    # ============= Plot =============
    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(time_days, displacement, marker='.', label='Observed', color='black')
    # prediction_plot = prediction[0] 
    # plt.plot(time_days, prediction_plot, marker='.', label='Predicted', color='blue')
    # plt.legend()
    # plt.title('Prediction vs Observation')

    # plt.subplot(2, 1, 2)
    # plt.plot(time_days, score, label='Anomaly score', color='red')
    # plt.legend()
    # plt.title('Residual-based anomaly magnitude')
    # plt.tight_layout()
    # plt.show()

    plot_results(time_days, displacement, mean_pred, mon) 


if __name__ == "__main__":
    main()
