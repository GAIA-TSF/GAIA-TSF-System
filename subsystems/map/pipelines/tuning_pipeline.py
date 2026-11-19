"""Hyperparameter tuning of TRF and LSTM by rolling-origin cross-validation.

Candidates come from ``tuning.search.<model>``. Each one is fitted and
scored on expanding-window chronological folds of the reference data,
excluding the configured test split. The best candidate has the lowest mean
MAE over leads and folds. It is then retrained by the normal learning
pipeline, so the test split still gives an unbiased final score.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import logging
from typing import Any

import numpy as np
import yaml

from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import ForecastSamples, rolling_origin_folds
from subsystems.map.pipelines.learning_pipeline import LearningPipeline, predict_values
from subsystems.map.utils.forecasts import lead_metrics, write_json


LOGGER = logging.getLogger(__name__)
TUNABLE = ('trf', 'lstm')


def candidate_grid(space: dict[str, list[Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Full grid of ``space``; a reproducible random subset above ``limit``."""
    if not space:
        raise ValueError('tuning.search for this model is empty.')
    for name, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'tuning.search value list for {name!r} must be non-empty.')
    names = list(space)
    grid = [dict(zip(names, values)) for values in itertools.product(*space.values())]
    if len(grid) > limit:
        chosen = np.random.default_rng(seed).choice(len(grid), size=limit, replace=False)
        grid = [grid[index] for index in sorted(chosen)]
    return grid


def apply_params(model_config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return ``model_config`` with dotted keys (``optimizer.lr``) overridden."""
    merged = copy.deepcopy(model_config)
    for dotted, value in params.items():
        target = merged
        *parents, key = dotted.split('.')
        for parent in parents:
            target = target.setdefault(parent, {})
            if not isinstance(target, dict):
                raise ValueError(f'Cannot set {dotted!r}: {parent!r} is not a mapping.')
        target[key] = value
    return merged


class TuningPipeline:
    """Select TRF or LSTM settings on chronological folds, then retrain."""

    def __init__(self, config: dict[str, Any], retrain: bool = True) -> None:
        self.config = config
        self.retrain = retrain
        self.learning = LearningPipeline(config)
        self.ctx = self.learning.ctx

    def run(self) -> dict[str, Any]:
        import subsystems.map.plugins.models  # noqa: F401

        ctx = self.ctx
        if ctx.model_name not in TUNABLE:
            raise ValueError(f'Tuning supports {", ".join(TUNABLE)}, not {ctx.model_name}.')
        settings = ctx.config.get('tuning') or {}
        folds_count = int(settings.get('folds', 3))
        space = (settings.get('search') or {}).get(ctx.model_name) or {}
        candidates = candidate_grid(space, int(settings.get('max_candidates', 20)), ctx.seed)

        loaded, windows, reference_mask, _ = self.learning.load_reference()
        folds_by_look_back: dict[int | None, list] = {}
        results = []
        for number, params in enumerate(candidates, start=1):
            model_config = apply_params(ctx.model_config, params)
            look_back = MODEL_REGISTRY[ctx.model_name](model_config, ctx.horizon).look_back()
            if look_back not in folds_by_look_back:
                samples, splits, _ = self.learning.build_splits(
                    loaded, reference_mask, windows, look_back
                )
                folds_by_look_back[look_back] = rolling_origin_folds(
                    self._without_test(samples, splits.test), folds_count
                )
            fold_rows = []
            for fold, (fit, validation) in enumerate(folds_by_look_back[look_back], start=1):
                model = MODEL_REGISTRY[ctx.model_name](model_config, ctx.horizon)
                model.set_random_seed(ctx.seed)
                model.train(fit.features, fit.changes)
                for row in lead_metrics(validation, predict_values(model, validation)):
                    fold_rows.append({'fold': fold, **row})
            score = float(np.mean([row['mae'] for row in fold_rows if row['mae'] is not None]))
            baseline = float(
                np.mean([row['constant_mae'] for row in fold_rows if row['constant_mae'] is not None])
            )
            results.append({'candidate': number, 'params': params, 'mean_mae': score,
                            'constant_mean_mae': baseline, 'folds': fold_rows})
            LOGGER.info('Candidate %d/%d %s: mean MAE %.6g (constant %.6g)',
                        number, len(candidates), params, score, baseline)

        best = min(results, key=lambda result: result['mean_mae'])
        best_config = apply_params(ctx.model_config, best['params'])
        output = ctx.results_dir / 'tuning' / ctx.experiment_name
        self._write_results(output, results)
        summary = {
            **ctx.identity(),
            'folds': folds_count,
            'score': 'mean MAE over leads and folds, in target_units',
            'candidate_count': len(candidates),
            'best_candidate': best['candidate'],
            'best_params': best['params'],
            'best_mean_mae': best['mean_mae'],
            'constant_mean_mae': best['constant_mean_mae'],
            'best_model_config': best_config,
        }
        write_json(output / 'best_params.json', summary)
        (output / 'tuned_config.yaml').write_text(
            '# Paste into subsystems/map/config.yaml to keep the tuned settings.\n'
            + yaml.safe_dump({'models': {ctx.model_name: best_config}}, sort_keys=False),
            encoding='utf-8',
        )
        LOGGER.info('Best %s settings: %s', ctx.model_name, best['params'])
        if self.retrain:
            tuned = copy.deepcopy(self.config)
            tuned.setdefault('models', {})[ctx.model_name] = best_config
            summary['learning'] = LearningPipeline(tuned).run()['metrics']
        return summary

    @staticmethod
    def _without_test(samples: ForecastSamples, test: ForecastSamples) -> ForecastSamples:
        """Drop the configured test rows (test origins x test pixels)."""
        held_out = np.isin(samples.origin_indices, np.unique(test.origin_indices)) & np.isin(
            samples.pixel_indices, np.unique(test.pixel_indices)
        )
        return samples.subset(~held_out)

    def _write_results(self, output: Any, results: list[dict[str, Any]]) -> None:
        """One row per candidate, fold, and lead."""
        identity = self.ctx.identity()
        rows = [
            {
                'kv': identity['kv'],
                'target_variable': identity['target_variable'],
                'target_units': identity['target_units'],
                'model': identity['model'],
                'candidate': result['candidate'],
                'params': json.dumps(result['params']),
                'candidate_mean_mae': result['mean_mae'],
                **row,
            }
            for result in results
            for row in result['folds']
        ]
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'tuning_results.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_tuning(config: dict[str, Any], retrain: bool = True) -> dict[str, Any]:
    """Functional tuning entry point."""
    return TuningPipeline(config, retrain).run()
