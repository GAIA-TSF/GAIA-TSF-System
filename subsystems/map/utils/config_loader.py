"""
YAML config loader

Purpose:
- Load experiment configuration from YAML
- Convert into object with attribute-style access

Why:
- Cleaner than dict["key"]
- Compatible with existing pipeline code
"""

import yaml


class Config:
    """
    Simple wrapper to allow dot access:
        config.variable instead of config["variable"]
    """

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            # Recursively convert nested dictionaries
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


def load_config(path='config.yaml'):
    """
    Load YAML file and return Config object
    """
    print(f'[Config] Loading config from {path}')

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)
