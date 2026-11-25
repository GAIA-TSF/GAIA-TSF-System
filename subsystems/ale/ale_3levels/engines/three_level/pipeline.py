"""Engine B pipeline: the prototype's monitoring run, moved from MAP (B1).

Moved from MAP ``pipelines/monitoring_pipeline.py`` (f11c5f2 + stash@{0}). The
sequence of computations in :meth:`ThreeLevelPipeline.run` is the prototype's,
unchanged. Only input and output changed:

* input: a ``monitoring_input``-shaped mapping built from the MAP -> ALE
  contract (:func:`inputs_from_contract`) instead of MAP's ``.npz`` hand-off;
* output: numerical arrays (``pixel_stacks.npz``) and JSON/CSV series instead
  of GeoTIFFs; dashboard, animation and the persistent-residual PNG are not
  produced here (plots will read saved results, plan step 9).

Calibration artifacts keep the prototype's create-once semantics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from ale_3levels.contract import CALIBRATION, ForecastContract
from ale_3levels.engines.three_level.anomaly_detection import (
    LegacyGlobalZScoreDetector,
    StatisticalAnomalyDetector,
)
from ale_3levels.engines.three_level.calibration import (
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    UNKNOWN,
    MissingCalibrationError,
)
from ale_3levels.engines.three_level.grid import GridDataset
from ale_3levels.engines.three_level.level1_zscore import (
    FORMULA,
    frozen_residual_calibration,
)
from ale_3levels.engines.three_level.level2_cusum import (
    Level2Parameters,
    compute_level2,
    frozen_level2_calibration,
    level2_metadata,
    pixel_level2,
)
from ale_3levels.engines.three_level.level3_regime import (
    Level3Parameters,
    compute_level3,
    frozen_level3_calibration,
    level3_metadata,
)
from ale_3levels.engines.three_level.quality import (
    QualityParameters,
    acquisition_quality,
    calibration_date_counts,
    run_quality_flags,
)
from ale_3levels.engines.three_level.series import (
    build_records,
    phases,
    write_series,
)
from ale_3levels.engines.three_level.temporal_monitoring import TemporalResidualMonitor
from ale_3levels.spatial import SpatialCoherenceDetector

LOGGER = logging.getLogger(__name__)

RATE_UNIT = 'mm/day'
# unified (B2.1): z = (e - mu_cal) / sigma_cal, frozen before monitoring.
# frozen_sigma: the prototype's z = e / sigma_cal (no centring).
# legacy_global: frozen_sigma levels plus the 950702d global z for comparison.
ZSCORE_FORMULAS = ('unified', 'frozen_sigma', 'legacy_global')
UNIFIED_FORMULA = 'z = (e - mu_cal) / sigma_cal'
# prototype: every pre-monitoring record (includes model-training dates);
# independent (B2.2): only records with split == calibration.
CALIBRATION_POPULATIONS = ('prototype', 'independent')
CALIBRATION_FILES = {
    'l1': 'l1_calibration.json',
    'l2': 'l2_calibration.json',
    'l3': 'l3_calibration.json',
}


def prototype_config(three_level: dict[str, Any]) -> dict[str, Any]:
    """Map ``three_level.yaml`` onto the configuration the moved classes read.

    The moved classes read the prototype's section names (``anomaly_detection``,
    ``monitoring.dashboard.*``). Keys set to ``null`` are omitted so that the
    prototype defaults apply exactly as before (e.g. ``level2.direction``).
    """

    def present(section: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in section.items() if value is not None}

    return {
        'anomaly_detection': present(three_level['level1']),
        'level2': present(three_level['level2']),
        'level3': present(three_level['level3']),
        'quality': present(three_level['quality']),
        'monitoring': {
            'dashboard': {
                'spatial_coherence': present(three_level['spatial_coherence']),
                'anomaly_magnitude_threshold': three_level[
                    'anomaly_magnitude_threshold'
                ],
                'acceleration_cusum': present(three_level['acceleration_cusum']),
                'regime': present(three_level['regime']),
            }
        },
    }


def inputs_from_contract(
    contract: ForecastContract, *, run_mode: str = UNKNOWN, run_mode_reason: str = ''
) -> dict[str, Any]:
    """The prototype's ``monitoring_input`` arrays from the hand-off contract.

    Rates in mm/day over the forecast interval; calibration = every date before
    the table's monitoring start, monitoring = the rest (prototype windows).
    """
    start = contract.monitoring_start_index
    if not 0 < start < contract.dates.size:
        raise ValueError('The monitoring start must split the acquisition axis.')
    return {
        'observed_stack': contract.observed_mm_per_day,
        'prediction_stack': contract.predicted_mm_per_day,
        'residual_stack': contract.residual_mm_per_day,
        'uncertainty_stack': np.empty((0, 0, 0), dtype=np.float64),
        'has_uncertainty': False,
        'mask': contract.mask,
        'fixed_support_mask': contract.mask.copy(),
        # B2.3: model identity checked against frozen calibrations.
        'model_id': contract.attrs['model_id'],
        'model_version': contract.attrs['model_version'],
        'lead': contract.lead,
        # B2.2: records never used for training/validation (split == calibration).
        'calibration_records': contract.split == CALIBRATION,
        'dates': tuple(str(value) for value in contract.dates),
        'target_dates': tuple(str(value) for value in contract.dates),
        'forecast_origin_dates': tuple(
            str(value) for value in contract.forecast_origin
        ),
        'source_label_dates': tuple(str(value) for value in contract.dates),
        'target_feature': f'{contract.attrs["target_feature"]}_rate_mm_per_day',
        'forecast_timing_schema_version': '1.0',
        'run_mode': run_mode,
        'run_mode_reason': run_mode_reason,
        'calibration_window': (0, start),
        'monitoring_window': (start, int(contract.dates.size)),
        'transform': contract.grid.transform,
        'crs': contract.grid.crs,
        'nodata': float('nan'),
    }


def model_identity(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """(model_id, model_version, lead) of the residuals, when the inputs carry it."""
    if not inputs.get('model_version'):
        return None
    return {
        'model_id': str(inputs['model_id']),
        'model_version': str(inputs['model_version']),
        'lead': int(inputs['lead']),
    }


def inputs_from_npz(path: Path) -> dict[str, Any]:
    """Read a prototype ``monitoring_input.npz`` (golden inputs) into a mapping."""
    with np.load(path, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    for name in (
        'dates',
        'target_dates',
        'forecast_origin_dates',
        'source_label_dates',
    ):
        if name in inputs:
            inputs[name] = tuple(str(value) for value in inputs[name])
    for name in (
        'target_feature',
        'forecast_timing_schema_version',
        'run_mode',
        'run_mode_reason',
        'crs',
    ):
        if name in inputs:
            inputs[name] = str(inputs[name].item())
    inputs['has_uncertainty'] = bool(inputs['has_uncertainty'].item())
    inputs['nodata'] = float(inputs['nodata'].item())
    inputs['transform'] = tuple(float(value) for value in inputs['transform'])
    inputs['calibration_window'] = tuple(int(v) for v in inputs['calibration_window'])
    inputs['monitoring_window'] = tuple(int(v) for v in inputs['monitoring_window'])
    return inputs


@dataclass(frozen=True)
class ThreeLevelRun:
    """Everything one engine-B run computed (arrays) and reported (metadata)."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    series_paths: tuple[Path, Path] | None
    # In-memory results (level objects, regions, windows) for the AlertResult
    # adapter; not written and not part of the prototype outputs.
    details: dict[str, Any] = field(default_factory=dict)


class ThreeLevelPipeline:
    """Run L1, L2 and L3 over one set of inputs and write the results."""

    def __init__(
        self,
        three_level: dict[str, Any],
        output_dir: Path,
        *,
        write_legacy: bool = True,
        calibration_dir: Path | None = None,
    ) -> None:
        formula = str(three_level.get('zscore_formula', 'unified'))
        if formula not in ZSCORE_FORMULAS:
            raise ValueError(f'zscore_formula must be one of {ZSCORE_FORMULAS}.')
        self.zscore_formula = formula
        calibration = three_level.get('calibration') or {}
        population = str(calibration.get('population', 'prototype'))
        if population not in CALIBRATION_POPULATIONS:
            raise ValueError(
                f'calibration.population must be one of {CALIBRATION_POPULATIONS}.'
            )
        if str(calibration.get('mode', 'pooled')) != 'pooled':
            raise ValueError('Only calibration.mode: pooled is implemented.')
        self.calibration_population = population
        self.minimum_calibration_dates = int(calibration.get('minimum_dates', 20))
        self.three_level = three_level
        self.config = prototype_config(three_level)
        self.output_root = Path(output_dir)
        # Prototype-format files (series JSON/CSV, pixel_stacks.npz, metadata).
        # The ALE run writes the common schema instead; the golden test keeps
        # them. Calibration artifacts are always written (engine state).
        self.write_legacy = write_legacy
        # B2.3: an external frozen calibration (from 'calibrate') must exist and
        # match the residuals' model identity; otherwise calibration is fitted
        # once into <output>/calibration.
        self.calibration_dir = (
            None if calibration_dir is None else Path(calibration_dir)
        )

    def run(self, inputs: dict[str, Any]) -> ThreeLevelRun:
        """Run the prototype's monitoring computation on ``inputs``."""
        dataset = GridDataset(
            dates=tuple(inputs['dates']),
            mask=np.asarray(inputs['mask'], dtype=bool),
            crs=str(inputs.get('crs', '')),
            transform=tuple(inputs.get('transform', ())),
            nodata=float(inputs.get('nodata', np.nan)),
            target_feature=str(inputs.get('target_feature', '')),
        )
        calibration = tuple(int(value) for value in inputs['calibration_window'])
        monitoring = tuple(int(value) for value in inputs['monitoring_window'])
        residual_stack = inputs['residual_stack']
        identity = model_identity(inputs)
        paths = self._calibration_paths(identity)
        calibration_residuals, calibration_dates, population = (
            self._calibration_population(
                inputs, residual_stack, dataset.mask, calibration, monitoring
            )
        )
        detector = StatisticalAnomalyDetector(self.config['anomaly_detection'])
        reused = [level for level, path in paths.items() if path.exists()]
        calibration_path = paths['l1']
        l1_calibration = frozen_residual_calibration(
            calibration_path,
            calibration_residuals,
            dataset.dates,
            calibration,
            monitoring[0],
            # Static TSF mask, as in the prototype.
            dataset.mask,
            unit=RATE_UNIT,
            target_feature=dataset.target_feature,
            population=population,
            identity=identity,
        )
        unified = self.zscore_formula == 'unified'
        # B2.1: centre every residual z on its frozen calibration mean.
        l1_center = (
            float(l1_calibration['mean_residual_diagnostic']) if unified else 0.0
        )
        level1 = {
            'formula': UNIFIED_FORMULA if unified else FORMULA,
            'threshold': detector.zscore_threshold,
            'direction': detector.direction,
            'comparison': 'strict',
            'calibration': l1_calibration,
        }
        if unified:
            level1['center'] = l1_center
        level2_parameters = Level2Parameters.from_config(self.config)
        l2_calibration_path = paths['l2']
        l2_calibration = frozen_level2_calibration(
            l2_calibration_path,
            calibration_residuals,
            dataset.dates,
            calibration,
            monitoring[0],
            dataset.mask,
            unit=RATE_UNIT,
            target_feature=dataset.target_feature,
            parameters=level2_parameters,
            population=population,
            identity=identity,
        )
        anomalies = detector.detect(
            dataset,
            residual_stack,
            persistence_start_time_index=monitoring[0],
            persistence_end_time_index=monitoring[1],
            calibration=l1_calibration,
            center=l1_center,
        )
        legacy = None
        if self.zscore_formula == 'legacy_global':
            legacy = LegacyGlobalZScoreDetector(
                dict(self.three_level['legacy_global'])
            ).detect(dataset, residual_stack, monitoring[0], monitoring[1])
        coherence = SpatialCoherenceDetector(self._spatial_coherence_config()).detect(
            anomalies.binary_stack,
            dataset.mask,
        )

        level2 = compute_level2(
            residual_stack,
            dataset.mask,
            calibration,
            monitoring,
            l2_calibration,
            level2_parameters,
            coherence.regions,
            calibration_dates=calibration_dates,
            center=float(l2_calibration['mean_residual']) if unified else 0.0,
            center_regions=unified,
        )
        level2_summary = level2_metadata(level2, dataset.dates)
        pixel_l2_maximum, pixel_l2_first_alarm = pixel_level2(
            residual_stack,
            l1_calibration['sigma'],
            dataset.mask,
            monitoring,
            level2_parameters,
            center=l1_center,
        )

        dashboard_config = self._dashboard_config()
        residual_monitor = TemporalResidualMonitor(dashboard_config)
        residual_monitor.calibration_dates = calibration_dates
        coherent_cusum = residual_monitor.spatial_coherent_acceleration_cusum(
            inputs['observed_stack'],
            dataset.dates,
            calibration,
            monitoring,
            coherence.binary_stack,
        )

        # L3 needs the temporal series.
        temporal = residual_monitor.analyze(
            observed_stack=inputs['observed_stack'],
            prediction_stack=inputs['prediction_stack'],
            dates=dataset.dates,
            calibration_window=calibration,
            monitoring_window=monitoring,
            uncertainty_stack=(
                inputs['uncertainty_stack'] if inputs['has_uncertainty'] else None
            ),
            fixed_support_mask=inputs['fixed_support_mask'],
            coherent_regions=coherence.regions,
        )
        l3_calibration_path = paths['l3']
        l3_calibration = frozen_level3_calibration(
            l3_calibration_path,
            temporal.regime_calibration,
            dataset.dates,
            unit=RATE_UNIT,
            target_feature=dataset.target_feature,
            identity=identity,
        )
        level3 = compute_level3(
            temporal.regime_change_score,
            temporal.regional_regime_change_score,
            level2,
            monitoring,
            Level3Parameters.from_config(self.config),
            coherence.regions,
            calibration=l3_calibration,
            calibration_scores=temporal.regime_calibration_score,
        )
        level3_summary = level3_metadata(level3, dataset.dates)

        quality_parameters = QualityParameters.from_config(self.config)
        quality_states, coverage = acquisition_quality(
            inputs['observed_stack'],
            inputs['prediction_stack'],
            dataset.mask,
            monitoring,
            quality_parameters,
        )
        run_mode = inputs.get('run_mode', UNKNOWN)
        quality_flags = run_quality_flags(
            run_mode=run_mode,
            l2_bias_warning=bool(level2_summary['bias_warning']),
            calibration_dates=calibration_date_counts(
                l1_calibration, l2_calibration, l3_calibration
            ),
            calibrations_reused=reused,
            parameters=quality_parameters,
        )
        timing = self._timing_columns(inputs, dataset.dates)
        series_paths = None
        if self.write_legacy:
            series_paths = write_series(
                self.output_root / 'monitoring_series',
                build_records(
                    {
                        **timing,
                        'phase': phases(len(dataset.dates), calibration, monitoring),
                        'quality_state': quality_states,
                        'observed_tsf_coverage': coverage,
                        **self._temporal_columns(temporal),
                        'l1_anomalous_pixels': self._pixel_counts(
                            anomalies.instantaneous_stack, len(dataset.dates)
                        ),
                        'l1_persistent_anomalous_pixels': self._pixel_counts(
                            anomalies.binary_stack, len(dataset.dates)
                        ),
                        'l1_coherent_anomalous_pixels': self._pixel_counts(
                            coherence.binary_stack, len(dataset.dates)
                        ),
                        'l2_z': level2.z,
                        'l2_instability_cusum': level2.instability_cusum,
                        'l2_overforecast_cusum': level2.overforecast_cusum,
                        'l2_alarm': level2.alarm,
                        'coherent_region_available': level2.regional_available,
                        'coherent_region_l2_instability_cusum': (
                            level2.regional_instability_cusum
                        ),
                        'coherent_region_l2_overforecast_cusum': (
                            level2.regional_overforecast_cusum
                        ),
                        'coherent_region_l2_alarm': level2.regional_alarm,
                        'regime_change_score': level3.score,
                        'coherent_region_regime_change_score': level3.regional_score,
                        'l3_level': level3.level,
                        **{
                            f'l3_{name}': flags
                            for name, flags in level3.criteria.items()
                        },
                    },
                    range(len(dataset.dates)),
                ),
                rate_unit=RATE_UNIT,
                metadata={
                    'run_mode': run_mode,
                    'target_feature': dataset.target_feature,
                    'calibration_window': list(calibration),
                    'monitoring_window': list(monitoring),
                },
            )

        arrays = {
            'residual_z': anomalies.zscore_stack,
            'anomaly_l1_binary': anomalies.instantaneous_stack,
            'anomaly_binary': anomalies.binary_stack,
            'anomaly_score': anomalies.score_stack,
            'anomaly_coherent': coherence.binary_stack,
            'level2_maximum_instability_cusum': pixel_l2_maximum,
            'level2_first_alarm_index': pixel_l2_first_alarm,
            'maximum_coherence_acceleration_cusum': coherent_cusum,
        }
        if legacy is not None:
            arrays['legacy_global_score'] = legacy.score_stack
            arrays['legacy_global_binary'] = legacy.binary_stack
        arrays_path = self.output_root / 'pixel_stacks.npz'
        if self.write_legacy:
            self.output_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(arrays_path, dates=np.asarray(dataset.dates), **arrays)

        result = {
            'schema_version': SCHEMA_VERSION,
            'contract_version': CONTRACT_VERSION,
            'engine': 'three_level',
            'zscore_formula': self.zscore_formula,
            'calibration_population': self.calibration_population,
            'calibration_identity': identity,
            'calibration_reused': reused,
            'n_calibration': self._n_calibration(
                calibration_dates, calibration, monitoring
            ),
            'n_calibration_records': (
                int(
                    np.isfinite(
                        calibration_residuals[: monitoring[0]][:, dataset.mask]
                    ).sum()
                )
            ),
            'minimum_calibration_dates': self.minimum_calibration_dates,
            'run_mode': run_mode,
            'run_mode_reason': inputs.get('run_mode_reason', ''),
            'lead_time': lead_time_summary(
                [
                    {
                        'lead_time_days': days,
                        'lead_time_acquisitions': acquisitions,
                    }
                    for days, acquisitions in zip(
                        timing['lead_time_days'], timing['lead_time_acquisitions']
                    )
                ]
            ),
            'quality': {
                'minimum_coverage': quality_parameters.minimum_coverage,
                'minimum_calibration_samples': (
                    quality_parameters.minimum_calibration_samples
                ),
                'state_counts_monitoring': {
                    state: int(
                        np.sum(quality_states[monitoring[0] : monitoring[1]] == state)
                    )
                    for state in (
                        'ok',
                        'warm_up',
                        'missing_acquisition',
                        'insufficient_coverage',
                    )
                },
                'run_quality_flags': quality_flags,
            },
            'monitoring_series_path': None
            if series_paths is None
            else str(series_paths[0]),
            'monitoring_series_csv_path': (
                None if series_paths is None else str(series_paths[1])
            ),
            'pixel_stacks_path': str(arrays_path) if self.write_legacy else None,
            'forecast_timing_schema_version': inputs.get(
                'forecast_timing_schema_version', 'legacy_unknown'
            ),
            'target_feature': inputs.get('target_feature', ''),
            'anomaly_summary': anomalies.summary,
            'legacy_global_summary': None if legacy is None else legacy.summary,
            'level1': level1,
            'l1_calibration_path': str(calibration_path),
            'level2': level2_summary,
            'l2_calibration_path': str(l2_calibration_path),
            'level3': level3_summary,
            'l3_calibration_path': str(l3_calibration_path),
            # Lookup for acquisition-index arrays such as the L2 first alarm.
            'acquisition_dates': list(dataset.dates),
            'spatial_coherence_summary': coherence.summary,
        }
        if self.write_legacy:
            write_json(self.output_root / 'monitoring_metadata.json', result)
        LOGGER.info('Engine three_level completed in %s', self.output_root)
        details = {
            'anomalies': anomalies,
            'coherence': coherence,
            'level2': level2,
            'level2_parameters': level2_parameters,
            'level3': level3,
            'l1_sigma': float(l1_calibration['sigma']),
            'l1_center': l1_center,
            'calibration_population': self.calibration_population,
            'n_calibration': self._n_calibration(
                calibration_dates, calibration, monitoring
            ),
            'calibration_window': calibration,
            'monitoring_window': monitoring,
        }
        return ThreeLevelRun(result, arrays, series_paths, details)

    def _calibration_population(
        self,
        inputs: dict[str, Any],
        residual_stack: np.ndarray,
        mask: np.ndarray,
        calibration: tuple[int, int],
        monitoring: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray | None, str]:
        """Residuals and dates that may estimate engine-B baselines.

        ``prototype``: the full residual stack (the calibration window is the
        whole pre-monitoring period, as in the prototype); no date mask.
        ``independent`` (B2.2): only records with ``split == calibration``,
        i.e. neither trained on nor used for validation/early stopping; other
        records are NaN for calibration. The date mask marks acquisitions with
        such records; TSF-level baselines (L3 regime, acceleration CUSUM,
        regional L2) use only those dates.
        """
        if self.calibration_population == 'prototype':
            return residual_stack, None, 'prototype'
        records = inputs.get('calibration_records')
        if records is None:
            raise ValueError(
                'calibration.population: independent needs calibration_records '
                '(split == calibration) in the inputs.'
            )
        records = np.asarray(records, dtype=bool) & mask[np.newaxis]
        dates = np.any(records, axis=(1, 2))
        dates[monitoring[0] :] = False
        n_dates = int(dates[calibration[0] : calibration[1]].sum())
        if n_dates < self.minimum_calibration_dates:
            LOGGER.warning(
                'Independent calibration has %d dates (< calibration.minimum_dates '
                '%d); engine-B baselines are uncertain.',
                n_dates,
                self.minimum_calibration_dates,
            )
        residuals = np.where(records, residual_stack, np.nan)
        return residuals, dates, 'calibration_split'

    def _n_calibration(
        self,
        dates: np.ndarray | None,
        calibration: tuple[int, int],
        monitoring: tuple[int, int],
    ) -> int:
        """Number of acquisitions that estimate the baselines."""
        if dates is None:
            return int(min(calibration[1], monitoring[0]) - calibration[0])
        return int(dates[calibration[0] : calibration[1]].sum())

    def _calibration_paths(self, identity: dict[str, Any] | None) -> dict[str, Path]:
        """Calibration files, keyed by model and lead (B2.3/B2.5).

        ``<root>/<model_id>/lead<k>/l*_calibration.json``; without a model
        identity (prototype inputs) ``<root>/l*_calibration.json``. An external
        ``calibration_dir`` must already hold all three files.
        """
        root = self.calibration_dir or self.output_root / 'calibration'
        if identity is not None:
            root = root / str(identity['model_id']) / f'lead{identity["lead"]}'
        paths = {level: root / name for level, name in CALIBRATION_FILES.items()}
        if self.calibration_dir is not None:
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise MissingCalibrationError(
                    f'No frozen calibration for this model and lead: {missing}. '
                    "Run 'calibrate' first."
                )
        return paths

    def _dashboard_config(self) -> dict[str, Any]:
        return self.config['monitoring']['dashboard']

    def _spatial_coherence_config(self) -> dict[str, Any]:
        return self._dashboard_config().get('spatial_coherence', {})

    @staticmethod
    def _pixel_counts(stack: np.ndarray | None, size: int) -> np.ndarray:
        """Flagged pixels per acquisition; NaN when the product is absent."""
        if stack is None:
            return np.full(size, np.nan)
        return np.count_nonzero(stack, axis=(1, 2))

    @staticmethod
    def _timing_columns(
        inputs: dict[str, Any], dates: tuple[str, ...]
    ) -> dict[str, list[Any]]:
        """Forecast origin and lead time per target date (contract 3.2)."""
        origins = inputs.get('forecast_origin_dates') or ('',) * len(dates)
        index = {value: position for position, value in enumerate(dates)}
        days: list[int | None] = []
        acquisitions: list[int | None] = []
        for position, (origin, target) in enumerate(zip(origins, dates)):
            if not origin:
                days.append(None)
                acquisitions.append(None)
                continue
            days.append((date.fromisoformat(target) - date.fromisoformat(origin)).days)
            acquisitions.append(position - index[origin] if origin in index else None)
        return {
            'date': list(dates),
            'forecast_origin_date': [origin or None for origin in origins],
            'lead_time_days': days,
            'lead_time_acquisitions': acquisitions,
        }

    @staticmethod
    def _temporal_columns(temporal: Any) -> dict[str, Any]:
        """Aggregate rates and supporting acceleration evidence, native units."""
        spread = temporal.uncertainty_mean
        return {
            'observed_mean_los_velocity': temporal.observed_mean,
            'predicted_baseline_velocity': temporal.predicted_mean,
            'prediction_spread': (
                np.full(temporal.observed_mean.size, np.nan)
                if spread is None
                else spread
            ),
            'mean_residual_rate': temporal.residual_mean,
            'acceleration_cusum_input': temporal.acceleration,
            'acceleration_cusum_instability': temporal.acceleration_cusum_instability,
            'acceleration_cusum_recovery': temporal.acceleration_cusum_recovery,
            'dynamics': temporal.dynamics,
            'oscillation': temporal.oscillation,
            'coherent_region_acceleration_cusum_instability': (
                temporal.regional_acceleration_cusum_instability
            ),
            'coherent_region_acceleration_cusum_recovery': (
                temporal.regional_acceleration_cusum_recovery
            ),
            'coherent_region_dynamics': np.where(
                temporal.regional_cusum_available, temporal.regional_dynamics, ''
            ),
        }


def lead_time_summary(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimum/maximum lead over rows that have a forecast origin.

    Moved from MAP ``utils/forecast_timing.py``.
    """

    def bounds(key: str) -> dict[str, int | None]:
        values = [row[key] for row in series if row.get(key) is not None]
        return {
            'min': min(values) if values else None,
            'max': max(values) if values else None,
        }

    return {
        'acquisitions': bounds('lead_time_acquisitions'),
        'days': bounds('lead_time_days'),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON artifact, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
