"""Issue forecasts over the monitored domain and match arriving observations."""

from __future__ import annotations

import csv
import json
import logging
from typing import Any

import numpy as np

from subsystems.map.core.registry import MODEL_REGISTRY
from subsystems.map.dataset import build_forecast_samples
from subsystems.map.pipelines.learning_pipeline import (
    load_kv_features,
    predict_values,
    resolve_windows,
)
from subsystems.map.utils.artifacts import write_error_by_lead, write_observed_vs_predicted
from subsystems.map.utils.config_loader import MapContext
from subsystems.map.utils.forecasts import (
    apply_interval,
    error_metrics,
    forecast_columns,
    grid_metadata,
    match_observations,
    read_table,
    row_keys,
    write_json,
    write_table,
)


LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    """Forecast every valid origin in ``mask_file``; export residuals."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.ctx = MapContext.from_config(config)

    def run(self) -> dict[str, Any]:
        import subsystems.map.plugins.models  # noqa: F401

        ctx = self.ctx
        experiment = self._trained_experiment()
        loaded = load_kv_features(ctx)
        windows = resolve_windows(ctx, loaded.dates)
        model = MODEL_REGISTRY[ctx.model_name].load(ctx.model_dir / 'model.pkl')
        samples = build_forecast_samples(
            loaded, ctx.feature_names, ctx.target_feature, ctx.horizon, model.look_back()
        )
        predicted = predict_values(model, samples)
        lower, upper = apply_interval(predicted, experiment['uncertainty'])
        forecasts = forecast_columns(
            samples, predicted, lower, upper, windows, ctx.acquisition_interval_days
        )
        self._mark_reference_rows(forecasts)

        metadata = {
            **ctx.identity(),
            'model_version': experiment['model_version'],
            'interval_level': experiment['uncertainty']['level'],
            'calibration_window': [loaded.dates[windows['calibration'][0]], loaded.dates[windows['calibration'][1] - 1]],
            'monitoring_window': [loaded.dates[windows['monitoring'][0]], loaded.dates[windows['monitoring'][1] - 1]],
            'latest_acquisition_date': loaded.dates[-1],
            **grid_metadata(samples),
        }
        output = ctx.forecast_dir
        forecast_path = write_table(
            output / 'forecasts', forecasts, {**metadata, 'product': 'map_forecasts'}
        )
        residuals = match_observations(
            forecasts, loaded.features[ctx.target_feature], loaded.dates, loaded.grid.width
        )
        residual_path = write_table(
            output / 'residuals', residuals, {**metadata, 'product': 'map_forecast_residuals'}
        )
        summary = self._error_summary(residuals)
        self._write_summary_csv(output / 'forecast_errors.csv', metadata, summary)
        if ctx.plotting:
            self._plots(residuals, summary)
        result = {
            **metadata,
            'forecast_table': str(forecast_path),
            'residual_table': str(residual_path),
            'forecast_rows': int(forecasts['lead'].size),
            'expected_target_rows': int(np.count_nonzero(forecasts['target_date_status'] == 'expected')),
            'matched_rows': int(residuals['lead'].size),
            'errors': summary,
        }
        write_json(output / 'inference.json', result)
        LOGGER.info('MAP forecasts written to %s', output)
        return result

    def _trained_experiment(self) -> dict[str, Any]:
        """Load training metadata and refuse a mismatched configuration."""
        path = self.ctx.model_dir / 'experiment.json'
        if not path.is_file():
            raise FileNotFoundError(
                f'{path} not found. Run MAP learning with the same dataset and model first.'
            )
        experiment = json.loads(path.read_text(encoding='utf-8'))
        expected = {**self.ctx.identity(), 'feature_names': self.ctx.feature_names}
        differing = [key for key, value in expected.items() if experiment.get(key) != value]
        if differing:
            raise ValueError(
                'Configuration differs from the trained model in: '
                + ', '.join(differing) + '. Re-run learning.'
            )
        return experiment

    def _mark_reference_rows(self, forecasts: dict[str, np.ndarray]) -> None:
        """Label rows that are reference train/validation/test samples."""
        path = self.ctx.model_dir / 'reference_forecasts'
        if not path.with_suffix('.npz').is_file():
            return
        reference = read_table(path)
        if reference['lead'].size == 0:
            return
        keys = row_keys(reference)
        order = np.argsort(keys)
        query = row_keys(forecasts)
        position = np.minimum(np.searchsorted(keys[order], query), keys.size - 1)
        hit = keys[order][position] == query
        forecasts['sample_role'][hit] = reference['sample_role'][order][position[hit]]

    def _error_summary(self, residuals: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        """Model and persistence errors by period and lead on matched rows."""
        summary = []
        for period in ('calibration', 'monitoring', 'outside', 'all'):
            selected = (
                np.ones(residuals['lead'].size, bool)
                if period == 'all'
                else residuals['period'] == period
            )
            for lead in range(1, self.ctx.horizon + 1):
                rows = selected & (residuals['lead'] == lead)
                if not np.any(rows):
                    continue
                observed = residuals['observed_value'][rows]
                model = error_metrics(observed, residuals['predicted_value'][rows])
                baseline = error_metrics(observed, residuals['persistence_value'][rows])
                summary.append(
                    {
                        'period': period,
                        'lead': lead,
                        **model,
                        **{f'constant_{k}': v for k, v in baseline.items() if k != 'valid_count'},
                    }
                )
        return summary

    def _plots(self, residuals: dict[str, np.ndarray], summary: list[dict[str, Any]]) -> None:
        """Monitoring-period observed-vs-predicted and error-by-lead diagnostics."""
        horizon = self.ctx.horizon
        units = str(self.ctx.dataset['target_units'])
        monitoring = residuals['period'] == 'monitoring'
        observed = np.full((monitoring.sum(), horizon), np.nan)
        predicted = np.full_like(observed, np.nan)
        lead = residuals['lead'][monitoring] - 1
        observed[np.arange(lead.size), lead] = residuals['observed_value'][monitoring]
        predicted[np.arange(lead.size), lead] = residuals['predicted_value'][monitoring]
        label = f'{self.ctx.experiment_name}: monitoring period'
        write_observed_vs_predicted(self.ctx.forecast_dir, observed, predicted, units, label)
        rows = [row for row in summary if row['period'] == 'monitoring']
        if rows:
            write_error_by_lead(self.ctx.forecast_dir, rows, units, label)

    @staticmethod
    def _write_summary_csv(path: Any, metadata: dict[str, Any], summary: list[dict[str, Any]]) -> None:
        keys = ('kv', 'target_variable', 'target_units', 'model', 'experiment', 'model_version')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            fields = [*keys, *(summary[0] if summary else ['period', 'lead'])]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({**{k: metadata[k] for k in keys}, **row} for row in summary)


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    """Functional inference entry point."""
    return InferencePipeline(config).run()
