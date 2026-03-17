
import json
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))
from subsystems.map.registry.model_registry import ModelRegistry


class TestModelRegistry:
    """
    Test suite for ML_R_06 – Model Registry.
    """

    def test_registry_file_creation(self):
        """
        Verify registry file is created if it does not exist.
        """

        with tempfile.TemporaryDirectory() as tmp:

            registry_path = Path(tmp) / 'model_registry.json'

            registry = ModelRegistry(registry_path)

            # assert registry_path.exists()
            assert registry is not None 

            with open(registry_path) as f:
                data = json.load(f)

            assert data == []

    def test_model_registration(self):
        """
        Verify a trained model can be registered.
        """

        with tempfile.TemporaryDirectory() as tmp:

            registry_path = Path(tmp) / 'model_registry.json'
            registry = ModelRegistry(registry_path)

            entry = registry.register_model(
                model_file='best_model.pt',
                dataset='synthetic',
                parameters={
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.1
                },
                monitoring_cfg={
                    'sigma_threshold': 2.5,
                    'persistence': 3
                },
                metrics={
                    'final_train_loss': 2.5,
                    'final_test_loss': 3.1
                }
            )

            assert entry['model_file'] == 'best_model.pt'
            assert entry['dataset'] == 'synthetic'
            assert 'metrics' in entry
            assert 'parameters' in entry
            assert 'timestamp' in entry

    def test_registry_persistence(self):
        """
        Verify registry correctly stores multiple models.
        """

        with tempfile.TemporaryDirectory() as tmp:

            registry_path = Path(tmp) / 'model_registry.json'
            registry = ModelRegistry(registry_path)

            registry.register_model(
                model_file='model1.pt',
                dataset='synthetic',
                parameters={'hidden_size': 32},
                monitoring_cfg={},
                metrics={'final_test_loss': 3.0}
            )

            registry.register_model(
                model_file='model2.pt',
                dataset='synthetic',
                parameters={'hidden_size': 64},
                monitoring_cfg={},
                metrics={'final_test_loss': 2.5}
            )

            with open(registry_path) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]['model_file'] == 'model1.pt'
            assert data[1]['model_file'] == 'model2.pt'

    def test_load_latest_model(self):
        """
        Verify latest model can be retrieved from registry.
        """

        with tempfile.TemporaryDirectory() as tmp:

            registry_path = Path(tmp) / 'model_registry.json'
            registry = ModelRegistry(registry_path)

            registry.register_model(
                model_file='model1.pt',
                dataset='synthetic',
                parameters={},
                monitoring_cfg={},
                metrics={}
            )

            registry.register_model(
                model_file='model2.pt',
                dataset='synthetic',
                parameters={},
                monitoring_cfg={},
                metrics={}
            )

            model_path, metadata = ModelRegistry.load_latest_model(tmp)

            assert metadata['model_file'] == 'model2.pt'
