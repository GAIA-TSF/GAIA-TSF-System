"""Post-training explainability artifacts for fitted tree ensembles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from subsystems.map.dataset.dataset_builder import Dataset
from subsystems.map.utils.artifacts import regression_metrics, write_json


LOGGER = logging.getLogger(__name__)


class _Regressor(Protocol):
    """Minimal fitted-estimator interface used for explanation."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict target values for tabular feature rows."""


def write_tree_explainability(
    output_dir: Path,
    model: object,
    validation_dataset: Dataset,
    config: dict[str, Any],
    random_seed: int,
    target_unit: str,
    target_value_scale: float,
) -> dict[str, Any]:
    """Write SHAP and grouped permutation artifacts for a tree model.

    SHAP uses a deterministic, temporally stratified validation sample. Grouped
    permutation importance is calculated on all validation samples, preserving
    the within-group structure while testing each group's incremental value.

    Args:
        output_dir: Experiment model-artifact directory.
        model: MAP predictive-model plugin containing a fitted tree estimator.
        validation_dataset: Held-out temporal validation samples.
        config: Top-level MAP explainability configuration.
        random_seed: Seed used for sampling and permutations.
        target_unit: Unit used on explanatory SHAP axes.
        target_value_scale: Native-to-display target conversion factor.

    Returns:
        Explainability metadata written to ``explainability.json``.

    Raises:
        ValueError: If enabled explainability configuration is invalid.
        RuntimeError: If SHAP is enabled but its optional dependency is missing.
    """
    if not bool(config.get('enabled', False)):
        return {'status': 'disabled'}
    if target_value_scale <= 0:
        raise ValueError('Explainability target_value_scale must be positive.')

    estimator = getattr(model, 'model', None)
    if estimator is None or not hasattr(estimator, 'estimators_'):
        metadata = {
            'status': 'not_supported',
            'reason': 'The selected model is not a fitted tree ensemble.',
        }
        write_json(output_dir / 'explainability.json', metadata)
        LOGGER.info('Skipping tree explainability: %s', metadata['reason'])
        return metadata
    if not hasattr(estimator, 'predict'):
        raise TypeError('Tree estimator does not implement predict().')

    shap_config = _mapping(config, 'shap')
    permutation_config = _mapping(config, 'grouped_permutation')
    sample_size = _positive_int(shap_config, 'sample_size')
    dependence_features = _string_list(shap_config, 'dependence_features')
    repeat_count = _positive_int(permutation_config, 'n_repeats')
    groups = _feature_groups(permutation_config)

    sample_indices = temporal_stratified_sample_indices(
        validation_dataset.time_indices,
        sample_size,
        random_seed,
    )
    sampled_features = validation_dataset.features[sample_indices]
    sampled_time_indices = validation_dataset.time_indices[sample_indices]
    feature_names = list(validation_dataset.feature_names)
    active_dependence_features = [
        name for name in dependence_features if name in validation_dataset.feature_names
    ]
    missing_dependence_features = sorted(
        set(dependence_features) - set(active_dependence_features),
    )

    shap_values = _tree_shap_values(estimator, sampled_features)
    if shap_values.shape != sampled_features.shape:
        raise ValueError(
            'TreeExplainer returned an unexpected SHAP shape: '
            f'{shap_values.shape}; expected {sampled_features.shape}.',
        )
    display_shap_values = shap_values * target_value_scale
    global_importance = _global_shap_importance(
        display_shap_values,
        feature_names,
    )
    _write_shap_global_plot(
        output_dir / 'shap_global_importance.png',
        global_importance,
        target_unit,
    )
    _write_shap_summary_plot(
        output_dir / 'shap_summary.png',
        display_shap_values,
        sampled_features,
        feature_names,
        target_unit,
    )
    for feature_name in active_dependence_features:
        _write_shap_dependence_plot(
            output_dir / f'shap_dependence_{feature_name}.png',
            display_shap_values,
            sampled_features,
            feature_names,
            feature_name,
            target_unit,
        )

    importance, skipped_groups = grouped_permutation_importance(
        estimator,
        validation_dataset.features,
        validation_dataset.targets,
        feature_names,
        groups,
        repeat_count,
        random_seed,
        target_value_scale,
    )
    _write_grouped_permutation_plot(
        output_dir / 'feature_importance_grouped.png',
        importance,
        target_unit,
    )
    metadata = {
        'status': 'completed',
        'evaluation_split': 'validation',
        'shap': {
            'explainer': 'TreeExplainer',
            'sample_count': int(sample_indices.size),
            'sampled_time_indices': sorted(
                int(value) for value in np.unique(sampled_time_indices)
            ),
            'global_importance': global_importance,
            'dependence_features': active_dependence_features,
            'missing_dependence_features': missing_dependence_features,
        },
        'grouped_permutation': {
            'n_repeats': repeat_count,
            'metric_unit': target_unit,
            'groups': importance,
            'skipped_groups': skipped_groups,
        },
    }
    write_json(output_dir / 'explainability.json', metadata)
    return metadata


def temporal_stratified_sample_indices(
    time_indices: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> np.ndarray:
    """Return a reproducible sample that represents each acquisition time.

    Args:
        time_indices: Temporal index of each candidate sample.
        sample_size: Maximum number of selected samples.
        random_seed: Seed for within-time sampling.

    Returns:
        Sorted sample-row indices, containing all rows when the requested size
        is not smaller than the input size.
    """
    values = np.asarray(time_indices)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('time_indices must be a non-empty one-dimensional array.')
    if sample_size < 1:
        raise ValueError('sample_size must be positive.')
    if sample_size >= values.size:
        return np.arange(values.size, dtype=np.int64)

    unique_times = np.unique(values)
    selected_times = unique_times
    if sample_size < unique_times.size:
        selected_times = unique_times[
            np.linspace(0, unique_times.size - 1, sample_size, dtype=np.int64)
        ]
    base_count, remainder = divmod(sample_size, selected_times.size)
    generator = np.random.default_rng(random_seed)
    selected: list[np.ndarray] = []
    for position, time_index in enumerate(selected_times):
        candidates = np.flatnonzero(values == time_index)
        requested = base_count + int(position < remainder)
        count = min(requested, candidates.size)
        selected.append(generator.choice(candidates, size=count, replace=False))

    result = np.sort(np.concatenate(selected))
    if result.size >= sample_size:
        return result[:sample_size]
    remaining = np.setdiff1d(np.arange(values.size), result, assume_unique=True)
    supplement = generator.choice(
        remaining,
        size=sample_size - result.size,
        replace=False,
    )
    return np.sort(np.concatenate((result, supplement)))


def grouped_permutation_importance(
    estimator: _Regressor,
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    configured_groups: dict[str, list[str]],
    n_repeats: int,
    random_seed: int,
    value_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Calculate held-out importance by permuting configured feature groups.

    A single row permutation is applied to every column in one group. This
    breaks its relationship to the target while retaining correlation within
    that group, such as the sine/cosine seasonal pair. Reported metrics are
    converted to display units with ``value_scale``.
    """
    if features.ndim != 2 or features.shape[0] != targets.size:
        raise ValueError('Features and targets must contain aligned 2D samples.')
    if n_repeats < 1:
        raise ValueError('n_repeats must be positive.')
    if value_scale <= 0:
        raise ValueError('value_scale must be positive.')
    if len(feature_names) != features.shape[1]:
        raise ValueError('feature_names do not match the feature matrix width.')

    baseline_metrics = regression_metrics(targets, estimator.predict(features))
    display_baseline_metrics = {
        metric: value * value_scale for metric, value in baseline_metrics.items()
    }
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    generator = np.random.default_rng(random_seed)
    importance: list[dict[str, Any]] = []
    skipped: dict[str, list[str]] = {}
    for group_name, names in configured_groups.items():
        active_names = [name for name in names if name in name_to_index]
        missing_names = [name for name in names if name not in name_to_index]
        if missing_names:
            skipped[group_name] = missing_names
        if not active_names:
            continue
        column_indices = [name_to_index[name] for name in active_names]
        rmse_increases: list[float] = []
        mae_increases: list[float] = []
        for _ in range(n_repeats):
            permuted = features.copy()
            permutation = generator.permutation(features.shape[0])
            permuted[:, column_indices] = features[permutation][:, column_indices]
            permuted_metrics = regression_metrics(targets, estimator.predict(permuted))
            rmse_increases.append(permuted_metrics['rmse'] - baseline_metrics['rmse'])
            mae_increases.append(permuted_metrics['mae'] - baseline_metrics['mae'])
        importance.append(
            {
                'group': group_name,
                'active_features': active_names,
                'rmse_increase_mean': float(np.mean(rmse_increases) * value_scale),
                'rmse_increase_std': float(np.std(rmse_increases) * value_scale),
                'mae_increase_mean': float(np.mean(mae_increases) * value_scale),
                'mae_increase_std': float(np.std(mae_increases) * value_scale),
                'baseline_metrics': display_baseline_metrics,
            },
        )
    importance.sort(key=lambda item: item['rmse_increase_mean'], reverse=True)
    return importance, skipped


def _tree_shap_values(estimator: _Regressor, features: np.ndarray) -> np.ndarray:
    """Return TreeExplainer values while keeping SHAP an optional import."""
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'SHAP explainability is enabled but the optional shap dependency is '
            'not installed. Install the project requirements and rerun learning.',
        ) from exc
    values = shap.TreeExplainer(estimator).shap_values(features)
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values, dtype=np.float64)


def _global_shap_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, float | str]]:
    """Summarize mean absolute SHAP magnitude per input feature."""
    magnitudes = np.mean(np.abs(shap_values), axis=0)
    ordering = np.argsort(magnitudes)[::-1]
    return [
        {
            'feature': feature_names[index],
            'mean_absolute_shap': float(magnitudes[index]),
        }
        for index in ordering
    ]


def _write_shap_global_plot(
    path: Path,
    importance: list[dict[str, float | str]],
    unit: str,
) -> None:
    """Write an ordered global SHAP magnitude bar chart."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [str(item['feature']) for item in reversed(importance)]
    values = [float(item['mean_absolute_shap']) for item in reversed(importance)]
    figure, axis = plt.subplots(figsize=(9, max(4, 0.5 * len(labels) + 1.5)))
    axis.barh(labels, values, color='tab:blue', alpha=0.8)
    suffix = f' [{unit}]' if unit else ''
    axis.set(
        xlabel=f'Mean |SHAP value|{suffix}',
        ylabel='Feature',
        title='Global SHAP feature importance',
    )
    axis.grid(axis='x', alpha=0.25)
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)


def _write_shap_summary_plot(
    path: Path,
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    unit: str,
) -> None:
    """Write a SHAP beeswarm summary plot."""
    import matplotlib
    import shap

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, max(5, 0.65 * len(feature_names) + 2)))
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        show=False,
    )
    axis = plt.gca()
    suffix = f' [{unit}]' if unit else ''
    axis.set_xlabel(f'Effect on predicted velocity{suffix}')
    axis.set_title('SHAP summary: feature effect on predicted velocity')
    plt.gcf().savefig(path, dpi=150, bbox_inches='tight')
    plt.close(plt.gcf())


def _write_shap_dependence_plot(
    path: Path,
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    feature_name: str,
    unit: str,
) -> None:
    """Write one SHAP dependence plot for a configured input feature."""
    import matplotlib
    import shap

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    shap.dependence_plot(
        feature_name,
        shap_values,
        features,
        feature_names=feature_names,
        interaction_index=None,
        show=False,
    )
    axis = plt.gca()
    suffix = f' [{unit}]' if unit else ''
    axis.set_ylabel(f'Effect on predicted velocity{suffix}')
    axis.set_title(f'SHAP dependence: {feature_name}')
    axis.grid(alpha=0.2)
    plt.gcf().savefig(path, dpi=150, bbox_inches='tight')
    plt.close(plt.gcf())


def _write_grouped_permutation_plot(
    path: Path,
    importance: list[dict[str, Any]],
    unit: str,
) -> None:
    """Write grouped permutation RMSE increases with repeat uncertainty."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [str(item['group']) for item in reversed(importance)]
    values = [float(item['rmse_increase_mean']) for item in reversed(importance)]
    errors = [float(item['rmse_increase_std']) for item in reversed(importance)]
    figure, axis = plt.subplots(figsize=(9, max(4, 0.7 * len(labels) + 1.5)))
    axis.barh(labels, values, xerr=errors, color='tab:green', alpha=0.8, capsize=3)
    suffix = f' [{unit}]' if unit else ''
    axis.set(
        xlabel=f'Validation RMSE increase after group permutation{suffix}',
        ylabel='Feature group',
        title='Grouped permutation importance',
    )
    axis.axvline(0.0, color='black', linewidth=0.8)
    axis.grid(axis='x', alpha=0.25)
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Read a required nested configuration mapping."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'explainability.{key} must be a mapping.')
    return value


def _positive_int(config: dict[str, Any], key: str) -> int:
    """Read a positive integer configuration value."""
    value = config.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f'explainability.{key} must be a positive integer.')
    return value


def _string_list(config: dict[str, Any], key: str) -> list[str]:
    """Read a list of non-empty strings from configuration."""
    value = config.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f'explainability.{key} must be a list of strings.')
    return value


def _feature_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    """Read configured feature groups while preserving configuration order."""
    value = config.get('groups')
    if not isinstance(value, dict):
        raise ValueError('explainability.grouped_permutation.groups must be a mapping.')
    groups: dict[str, list[str]] = {}
    for name, members in value.items():
        if not isinstance(name, str) or not isinstance(members, list):
            raise ValueError('Each explainability feature group must be a string list.')
        groups[name] = [str(member) for member in members]
    return groups
