
import yaml 
import torch

"""Utils for MAp module. 
"""

def _load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _select_device(device_config: str) -> torch.device:
    if device_config == 'auto':
        return torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
    return torch.device(device_config)
