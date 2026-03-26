import os
import json
from datetime import datetime


def update_experiment_index(root_dir, exp_name, metrics):
    """
    Update the global experiment index with the latest experiment results.
    """

    index_file = os.path.join(root_dir, 'experiments_index.json')

    # load existing index
    index = []
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)

    entry = {'experiment': exp_name, 'best_test_loss': metrics['best_test_loss']}

    index.append(entry)

    # write updated index
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)


class ExperimentTracker:
    """
    Lightweight MLflow-like experiment tracker.
    """

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)

        self.params_file = os.path.join(exp_dir, 'params.json')
        self.metrics_file = os.path.join(exp_dir, 'metrics.json')
        self.exp_file = os.path.join(exp_dir, 'experiment.json')
        self.artifacts_file = os.path.join(exp_dir, 'artifacts.json')

    # -------------------------------------------------
    # start experiment
    # -------------------------------------------------
    def start(self, dataset, config):
        exp_data = {
            'dataset': dataset,
            'start_time': datetime.utcnow().isoformat(),
        }

        with open(self.exp_file, 'w') as f:
            json.dump(exp_data, f, indent=2)

        with open(self.params_file, 'w') as f:
            json.dump(config, f, indent=2)

    # -------------------------------------------------
    # log metrics
    # -------------------------------------------------
    def log_metrics(self, train_losses, test_losses):
        metrics = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': float(train_losses[-1]),
            'final_test_loss': float(test_losses[-1]),
            'best_test_loss': float(min(test_losses)),
        }

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    # -------------------------------------------------
    # log artifact
    # -------------------------------------------------
    def log_artifact(self, path):
        artifacts = []

        if os.path.exists(self.artifacts_file):
            with open(self.artifacts_file) as f:
                artifacts = json.load(f)

        artifacts.append(path)

        with open(self.artifacts_file, 'w') as f:
            json.dump(artifacts, f, indent=2)

    # -------------------------------------------------
    # finish experiment
    # -------------------------------------------------
    def finish(self):
        with open(self.exp_file) as f:
            exp_data = json.load(f)

        exp_data['end_time'] = datetime.utcnow().isoformat()

        with open(self.exp_file, 'w') as f:
            json.dump(exp_data, f, indent=2)
