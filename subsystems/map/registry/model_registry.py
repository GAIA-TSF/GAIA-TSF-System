import json
import os
from datetime import datetime


class ModelRegistry:
    def __init__(self, registry_path):
        self.registry_path = registry_path

        # ensure directory exists
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)

        if not os.path.exists(registry_path):
            with open(registry_path, 'w') as f:
                json.dump([], f)

    def register_model(
        self,
        model_file,
        dataset,
        parameters,
        monitoring_cfg,
        metrics,
        version='1.0',
    ):
        with open(self.registry_path) as f:
            registry = json.load(f)

        model_id = f'model_{len(registry) + 1}'

        entry = {
            'model_id': model_id,
            'version': version,
            'model_file': model_file,
            'dataset': dataset,
            'parameters': parameters,
            'monitoring': monitoring_cfg,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat(),
        }

        registry.append(entry)

        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        return entry

    @staticmethod
    def load_latest_model(exp_dir):
        """
        Load latest registered model from registry.
        Returns model_path and registry metadata.
        """

        registry_file = os.path.join(exp_dir, 'model_registry.json')

        if not os.path.exists(registry_file):
            raise RuntimeError(f'No model registry found in {exp_dir}')

        with open(registry_file) as f:
            registry = json.load(f)

        if len(registry) == 0:
            raise RuntimeError('Model registry is empty')

        latest = registry[-1]

        model_path = os.path.join(exp_dir, latest['model_file'])

        return model_path, latest
