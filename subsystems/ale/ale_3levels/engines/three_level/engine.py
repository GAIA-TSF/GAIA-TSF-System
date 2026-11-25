"""Engine B, ``three_level``: the moved prototype behind the common interface.

:class:`ThreeLevelEngine` runs :class:`ThreeLevelPipeline` unchanged and
converts its results into the shared :class:`AlertResult` without altering
any engine-B number. Level per pixel and acquisition (monitoring dates only):

* ``L1`` (pixel scope): the pixel's instantaneous L1 flag, strict directional
  ``|z| > zscore_threshold``.
* ``L2`` (region scope, ``region_id`` 0 = whole TSF): the TSF-wide L2 alarm,
  or an L3 ``watch`` from a sustained regime score.
* ``L2`` (region scope, ``region_id`` k): the regional L2 alarm, for pixels of
  the coherence-qualified regions active at that date. The prototype keeps
  one combined regional series (any region alarming), so every active region
  inherits it.
* ``L3`` (region scope, ``region_id`` k): L3 ``alert``, for pixels of the
  coherence-qualified regions active at that date. The prototype decides L3
  once for the whole TSF (score = max of TSF and regional scores; alert
  requires that some coherent region exists); ALE attributes it to those
  regions (decision 2026-11-25). Other TSF pixels keep their lower level.
* ``NONE`` otherwise; ``NO_DATA`` before monitoring, outside the TSF mask or
  where the pixel has no residual.

The highest applicable level wins; its scope and region are reported.
Metric values: ``z`` (pixel), ``pixel_cusum_pos``/``pixel_cusum_neg`` (pixel
CUSUM of z, diagnostic), ``cusum_pos``/``cusum_neg`` (the TSF-wide L2 CUSUM
that drives L2), ``regional_cusum_pos``/``regional_cusum_neg`` (combined
regional series) and ``regime_change_score`` (the L3 evidence score), the
last three broadcast to the pixels of their region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ale_3levels import __version__
from ale_3levels.config import config_hash
from ale_3levels.contract import ForecastContract
from ale_3levels.engines.base import (
    NO_DATA,
    NO_REGION,
    PIXEL_SCOPE,
    REGION_SCOPE,
    AlertResult,
    severity_from_levels,
)
from ale_3levels.engines.three_level.calibration import UNKNOWN
from ale_3levels.engines.three_level.pipeline import (
    ThreeLevelPipeline,
    ThreeLevelRun,
    inputs_from_contract,
)

ENGINE_VERSION = f'{__version__}+three_level.prototype'
LEVELS = (NO_DATA, 'NONE', 'L1', 'L2', 'L3')
NO_DATA_CODE, NONE, L1, L2, L3 = range(5)
TRIGGER_REASONS = (
    'no_data:not_monitored',
    'no_data:outside_mask',
    'no_data:missing_residual',
    'none',
    'l1:pixel_z',
    'l2:tsf_cusum',
    'l2:regime_watch_score',
    'l2:region_cusum',
    'l3:alert',
)
TSF_REGION = 0


class ThreeLevelEngine:
    """Engine B: L1 z-score, L2 CUSUM, L3 regime-change evidence."""

    name = 'three_level'
    version = ENGINE_VERSION

    def __init__(
        self,
        config: dict[str, Any],
        work_dir: Path,
        *,
        run_mode: str = UNKNOWN,
        calibration_dir: Path | None = None,
    ) -> None:
        self.raw_config = dict(config)
        self.config_hash = config_hash(self.raw_config)
        self.work_dir = Path(work_dir)
        self.run_mode = run_mode
        # calibration_dir: a frozen calibration (from 'calibrate'); its model
        # identity must match the residuals (B2.3). None: fit into work_dir.
        self.pipeline = ThreeLevelPipeline(
            self.raw_config,
            self.work_dir,
            write_legacy=False,
            calibration_dir=calibration_dir,
        )
        self.last_run: ThreeLevelRun | None = None
        self.last_state = None

    def fit(self, calibration_data: ForecastContract) -> None:
        """Calibration is frozen inside the run (prototype behaviour, pre-B2)."""

    def score(
        self, monitoring_data: ForecastContract, state: Any | None = None
    ) -> AlertResult:
        """Run the prototype pipeline and express its result per pixel."""
        inputs = inputs_from_contract(monitoring_data, run_mode=self.run_mode)
        run = self.pipeline.run(inputs, state)
        self.last_run = run
        self.last_state = run.state
        return to_alert_result(
            run,
            monitoring_data,
            engine_version=self.version,
            config_hash=self.config_hash,
            recommended_actions=self.raw_config.get('recommended_actions') or {},
        )


def to_alert_result(
    run: ThreeLevelRun,
    contract: ForecastContract,
    *,
    engine_version: str,
    config_hash: str,
    recommended_actions: dict[str, list[str]],
) -> AlertResult:
    """Convert one pipeline run into the common schema (numbers unchanged)."""
    details = run.details
    level2, level3 = details['level2'], details['level3']
    regions = details['coherence'].regions
    start, end = details['monitoring_window']
    z = run.arrays['residual_z']
    l1_flag = run.arrays['anomaly_l1_binary']
    shape = z.shape
    time_count = shape[0]
    mask = np.broadcast_to(contract.mask, shape)
    monitored = np.zeros(time_count, dtype=bool)
    monitored[start:end] = True
    monitored3 = np.broadcast_to(monitored[:, None, None], shape)

    # Region index map: lowest region k (1-based) containing each pixel, per date.
    region_map = np.full(shape, NO_REGION, dtype=np.int32)
    for number, region in reversed(list(enumerate(regions, start=1))):
        active = np.arange(time_count) >= region.activation_index
        cells = active[:, None, None] & region.support[None]
        region_map[cells] = number

    tsf_alarm = monitored & level2.alarm
    regional_alarm = monitored & level2.regional_alarm
    l3_alert = level3.level == 'alert'
    l3_watch_score = monitored & level3.criteria['sustained_watch_score']

    def series(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[:, None, None], shape)

    region_level = np.full(shape, NONE, dtype=np.int8)
    region_reason = np.full(shape, TRIGGER_REASONS.index('none'), dtype=np.int16)
    region_id = np.full(shape, NO_REGION, dtype=np.int32)
    in_region = region_map >= 1
    regional = series(regional_alarm) & in_region
    region_level[regional] = L2
    region_reason[regional] = TRIGGER_REASONS.index('l2:region_cusum')
    region_id[regional] = region_map[regional]
    for flags, level, reason in (
        (l3_watch_score, L2, 'l2:regime_watch_score'),
        (tsf_alarm, L2, 'l2:tsf_cusum'),
    ):
        cells = series(flags) & mask & (level >= region_level)
        region_level[cells] = level
        region_reason[cells] = TRIGGER_REASONS.index(reason)
        region_id[cells] = TSF_REGION
    # L3 is attributed to the coherence-qualified regions active at the date
    # (an L3 alert requires one); the rest of the TSF keeps its lower level.
    cells = series(l3_alert) & mask & in_region
    region_level[cells] = L3
    region_reason[cells] = TRIGGER_REASONS.index('l3:alert')
    region_id[cells] = region_map[cells]

    level = np.full(shape, NONE, dtype=np.int8)
    reason = np.full(shape, TRIGGER_REASONS.index('none'), dtype=np.int16)
    scope = np.full(shape, PIXEL_SCOPE, dtype=np.int8)
    pixel_l1 = l1_flag.astype(bool)
    level[pixel_l1] = L1
    reason[pixel_l1] = TRIGGER_REASONS.index('l1:pixel_z')
    from_region = region_level > level
    level[from_region] = region_level[from_region]
    reason[from_region] = region_reason[from_region]
    scope[from_region] = REGION_SCOPE
    region_out = np.where(from_region, region_id, NO_REGION).astype(np.int32)

    no_residual = ~np.isfinite(z)
    for cells, cause in (
        (mask & monitored3 & no_residual, 'no_data:missing_residual'),
        (~mask, 'no_data:outside_mask'),
        (~monitored3, 'no_data:not_monitored'),
    ):
        level[cells] = NO_DATA_CODE
        reason[cells] = TRIGGER_REASONS.index(cause)
        scope[cells] = PIXEL_SCOPE
        region_out[cells] = NO_REGION

    pixel_pos, pixel_neg = details['pixel_cusum']
    tsf = np.where(mask, 1.0, np.nan)
    regional_cells = np.where(in_region, 1.0, np.nan)
    metric_values = {
        'z': z,
        'pixel_cusum_pos': pixel_pos,
        'pixel_cusum_neg': pixel_neg,
        'cusum_pos': series(level2.instability_cusum) * tsf,
        'cusum_neg': series(level2.overforecast_cusum) * tsf,
        'regional_cusum_pos': series(level2.regional_instability_cusum)
        * regional_cells,
        'regional_cusum_neg': series(level2.regional_overforecast_cusum)
        * regional_cells,
        'regime_change_score': series(level3.evidence_score) * tsf,
    }
    metric_values = {
        name: np.array(values, dtype=np.float64)
        for name, values in metric_values.items()
    }
    return AlertResult(
        engine='three_level',
        engine_version=engine_version,
        config_hash=config_hash,
        dates=np.asarray(contract.dates),
        levels=LEVELS,
        level=level,
        severity=severity_from_levels('three_level', LEVELS, level),
        metric_values=metric_values,
        trigger_reasons=TRIGGER_REASONS,
        trigger_reason=reason,
        recommended_actions={
            str(key): [str(text) for text in value or []]
            for key, value in recommended_actions.items()
        },
        attrs={
            'zscore_formula': run.metadata['zscore_formula'],
            'monitoring_window': [int(start), int(end)],
            'calibration_window': list(details['calibration_window']),
            'l1_sigma': details['l1_sigma'],
            'l1_center': details['l1_center'],
            'calibration_population': details['calibration_population'],
            'n_calibration': details['n_calibration'],
            'calibration_identity': run.metadata.get('calibration_identity'),
            'calibration_reused': run.metadata.get('calibration_reused'),
            'resets': run.metadata.get('resets'),
            'reset_after_l3': run.metadata.get('reset_after_l3'),
            'continued_from_state': run.metadata.get('continued_from_state'),
            'regions': [
                {
                    'region_id': number,
                    'activation_date': str(contract.dates[region.activation_index]),
                    'pixels': int(region.support.sum()),
                }
                for number, region in enumerate(regions, start=1)
            ],
            'level_rule': (
                'L1 pixel z; L2 TSF alarm or L3 watch score (region 0) or '
                'regional alarm (active coherent regions); L3 alert (active '
                'coherent regions only)'
            ),
            'l2_decision_threshold': level2.parameters.decision_threshold,
            'l3_watch_threshold': level3.parameters.watch_threshold,
            'l3_alert_threshold': level3.parameters.alert_threshold,
            'first_l2_alarm_date': run.metadata['level2']['first_alarm_date'],
            'first_l3_alert_date': run.metadata['level3']['first_alert_date'],
        },
        level_scope=scope,
        region_id=region_out,
    )
