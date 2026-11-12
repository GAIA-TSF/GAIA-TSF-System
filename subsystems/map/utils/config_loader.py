"""MAP configuration loading and the resolved per-run context."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import yaml


LOGGER = logging.getLogger(__name__)
MODELS = ('constant', 'trf', 'lstm')
HORIZONS = (1, 3, 5)
_SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
_LAGGED = re.compile(r'(^|_)lag\d+$')
# Alert-era keys that MAP no longer reads; fail loudly instead of ignoring them.
_REMOVED_KEYS = ('anomaly_detection', 'monitoring', 'baseline_model', 'variable', 'variables')


def load_config(
    path: str | Path,
    dataset: str | None = None,
    model: str | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Load a MAP YAML document and apply optional command-line selections."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('MAP configuration must contain a YAML mapping.')
    config['_config_path'] = str(config_path)
    for key, value in (('dataset', dataset), ('model', model), ('experiment', experiment)):
        if value is not None:
            config[key] = value
    LOGGER.info('Loaded MAP configuration from %s', config_path)
    return config


@dataclass(frozen=True)
class MapContext:
    """Validated settings for one KV dataset, model, and experiment."""

    config: dict[str, Any]
    dataset_name: str
    dataset: dict[str, Any]
    model_name: str
    experiment_name: str
    experiment_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'MapContext':
        removed = [key for key in _REMOVED_KEYS if key in config]
        if removed:
            raise KeyError(
                'MAP is forecasting-only; remove obsolete configuration keys: '
                + ', '.join(removed)
            )
        dataset_name = _string(config, 'dataset')
        dataset = config.get('datasets', {}).get(dataset_name)
        if not isinstance(dataset, dict):
            raise KeyError(f'Missing configuration: datasets.{dataset_name}')
        for key in ('kv', 'experiment_dir', 'mask_file', 'target_feature',
                    'target_variable', 'target_units', 'features'):
            if dataset.get(key) in (None, '', []):
                raise KeyError(f'datasets.{dataset_name}.{key} is required.')
        if _LAGGED.search(str(dataset['target_feature'])):
            raise ValueError(
                f"Target {dataset['target_feature']!r} is a lagged stack. Forecast the "
                'unlagged target; lagged stacks may only be predictors.'
            )
        model_name = _string(config, 'model')
        if model_name not in MODELS:
            raise ValueError(f'model must be one of: {", ".join(MODELS)}')
        experiment_name = str(config.get('experiment') or f"{dataset['kv']}_{model_name}")
        if not _SAFE_NAME.fullmatch(experiment_name):
            raise ValueError('experiment must contain only letters, digits, ".", "_", "-".')
        horizon = config.get('forecast', {}).get('horizon')
        if horizon not in HORIZONS:
            raise ValueError(f'forecast.horizon must be one of {HORIZONS}.')
        config_path = Path(str(config['_config_path']))
        root = Path(str(dataset['experiment_dir'])).expanduser()
        if not root.is_absolute():
            root = (config_path.parent / root).resolve()
        return cls(config, dataset_name, dataset, model_name, experiment_name, root)

    @property
    def kv(self) -> str:
        return str(self.dataset['kv'])

    @property
    def target_feature(self) -> str:
        return str(self.dataset['target_feature'])

    @property
    def feature_names(self) -> list[str]:
        return [str(name) for name in self.dataset['features']]

    @property
    def horizon(self) -> int:
        return int(self.config['forecast']['horizon'])

    @property
    def interval_level(self) -> float:
        level = float(self.config['forecast'].get('interval_level', 0.9))
        if not 0.0 < level < 1.0:
            raise ValueError('forecast.interval_level must be in (0, 1).')
        return level

    @property
    def acquisition_interval_days(self) -> float | None:
        value = self.config['forecast'].get('acquisition_interval_days')
        return None if value is None else float(value)

    @property
    def model_config(self) -> dict[str, Any]:
        value = self.config.get('models', {}).get(self.model_name) or {}
        if not isinstance(value, dict):
            raise ValueError(f'models.{self.model_name} must be a mapping.')
        return value

    @property
    def seed(self) -> int:
        return int(self.config.get('seed', 42))

    @property
    def plotting(self) -> bool:
        return bool(self.config.get('plotting', {}).get('enabled', False))

    @property
    def temporal_alignment_method(self) -> str:
        return str(self.config.get('data', {}).get('temporal_alignment_method', 'exact'))

    @property
    def results_dir(self) -> Path:
        return self.experiment_dir / 'results'

    @property
    def model_dir(self) -> Path:
        return self.results_dir / 'models' / self.experiment_name

    @property
    def forecast_dir(self) -> Path:
        return self.results_dir / 'forecasts' / self.experiment_name

    @property
    def feature_dirs(self) -> list[Path]:
        """Conventional DAG output directories searched for feature stems."""
        return [self.results_dir / name for name in ('features', 'temporal_features', 'meteo_features')]

    def static_file(self, filename: object) -> Path:
        """Resolve a file name relative to ``<experiment_dir>/static``."""
        path = Path(str(filename)).expanduser()
        return path if path.is_absolute() else self.experiment_dir / 'static' / path

    def identity(self) -> dict[str, Any]:
        """Fields shared by every MAP forecast product."""
        return {
            'kv': self.kv,
            'dataset': self.dataset_name,
            'model': self.model_name,
            'experiment': self.experiment_name,
            'target_variable': str(self.dataset['target_variable']),
            'target_units': str(self.dataset['target_units']),
            'target_feature': self.target_feature,
            'horizon': self.horizon,
        }


def _string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise KeyError(f'Missing MAP configuration key: {key}')
    return value
