"""Tests for the L3 regime-change evidence contract."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ale_3levels.engines.three_level.level2_cusum import Level2Parameters, Level2Result
from ale_3levels.engines.three_level.level3_regime import (
    Level3Parameters,
    compute_level3,
    frozen_level3_calibration,
    level3_metadata,
    regime_calibration_replay,
    sustained,
    z_equivalent,
)
from ale_3levels.engines.three_level.temporal_monitoring import TemporalResidualMonitor
from ale_3levels.spatial import SpatialCoherenceRegion

PARAMETERS = Level3Parameters(
    watch_threshold=0.4, alert_threshold=0.7, consecutive_acquisitions=2
)


def _level2(
    alarm: np.ndarray, regional_alarm: np.ndarray | None = None
) -> Level2Result:
    size = alarm.size
    nan = np.full(size, np.nan)
    return Level2Result(
        z=nan,
        instability_cusum=nan,
        overforecast_cusum=nan,
        alarm=alarm,
        first_alarm_index=None,
        regional_available=np.zeros(size, dtype=bool),
        regional_instability_cusum=nan,
        regional_overforecast_cusum=nan,
        regional_alarm=(
            np.zeros(size, dtype=bool) if regional_alarm is None else regional_alarm
        ),
        regional_first_alarm_index=None,
        regional_sigmas=(),
        parameters=Level2Parameters(0.5, 5.0, 'negative'),
        calibration={},
    )


def _region(activation: int, shape=(2, 2)) -> SpatialCoherenceRegion:
    return SpatialCoherenceRegion(
        support=np.ones(shape, dtype=bool), activation_index=activation
    )


def test_sustained_requires_consecutive_inclusive_and_nan_breaks_run() -> None:
    values = np.array([0.4, 0.39, 0.4, 0.4, np.nan, 0.5, 0.5, 0.5])

    assert sustained(values, 0.4, 2).tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
    ]
    assert sustained(values, 0.4, 1)[0]


def test_score_alone_never_raises_alert() -> None:
    score = np.full(8, 0.95)
    no_alarm = np.zeros(8, dtype=bool)

    result = compute_level3(
        score,
        np.full(8, np.nan),
        _level2(no_alarm),
        (2, 8),
        PARAMETERS,
        (_region(2),),
    )

    assert result.level[:2].tolist() == ['not_evaluated'] * 2
    # First monitoring date has only one acquisition above threshold.
    assert result.level[2:].tolist() == ['normal'] + ['watch'] * 5
    assert result.first_alert_index is None
    assert result.first_watch_index == 3


def test_alert_requires_l2_sustained_score_and_coherent_region() -> None:
    score = np.array([0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    alarm = np.array([False, False, False, False, True, True, True, True])

    without_region = compute_level3(
        score, np.full(8, np.nan), _level2(alarm), (2, 8), PARAMETERS
    )
    with_region = compute_level3(
        score,
        np.full(8, np.nan),
        _level2(alarm),
        (2, 8),
        PARAMETERS,
        (_region(5),),
    )

    assert 'alert' not in without_region.level
    assert with_region.level.tolist() == [
        'not_evaluated',
        'not_evaluated',
        'normal',
        'watch',
        'watch',
        'alert',
        'alert',
        'alert',
    ]
    assert with_region.first_alert_index == 5
    assert with_region.criteria['coherent_region'][5]


def test_l2_alarm_alone_is_watch_and_regional_score_counts() -> None:
    alarm = np.zeros(6, dtype=bool)
    regional_alarm = np.array([False, False, True, True, True, True])
    regional = np.array([np.nan, np.nan, 0.75, 0.75, 0.75, 0.2])

    result = compute_level3(
        np.zeros(6),
        regional,
        _level2(alarm, regional_alarm),
        (1, 6),
        PARAMETERS,
        (_region(2),),
    )

    assert result.level.tolist() == [
        'not_evaluated',
        'normal',
        'watch',
        'alert',
        'alert',
        'watch',
    ]


def test_parameters_validate_and_read_deprecated_dashboard_keys(caplog) -> None:
    with pytest.raises(ValueError, match='watch_threshold must be below'):
        Level3Parameters(0.7, 0.4)
    with pytest.raises(ValueError, match='consecutive_acquisitions'):
        Level3Parameters(consecutive_acquisitions=0)
    legacy = {
        'monitoring': {
            'dashboard': {
                'regime': {'medium_risk_threshold': 0.3, 'high_risk_threshold': 0.8}
            }
        },
        'level3': {'consecutive_acquisitions': 3},
    }

    parameters = Level3Parameters.from_config(legacy)

    assert parameters == Level3Parameters(0.3, 0.8, 3)
    assert 'deprecated' in caplog.text
    assert Level3Parameters.from_config({}) == PARAMETERS


def test_threshold_z_equivalent_inverts_the_score_map() -> None:
    assert z_equivalent(0.7) == pytest.approx(-np.log(0.3))
    assert 1.0 - np.exp(-z_equivalent(0.4)) == pytest.approx(0.4)


def test_calibration_replay_counts_sustained_score_episodes() -> None:
    scores = np.array([np.nan, 0.1, 0.5, 0.5, 0.1, 0.8, 0.8, 0.8, 0.1])

    replay = regime_calibration_replay(scores, PARAMETERS)

    assert replay['samples'] == 8
    assert replay['sustained_watch_score_episodes'] == 2
    assert replay['sustained_alert_score_episodes'] == 1
    assert replay['fraction_at_or_above_alert'] == pytest.approx(3 / 8)


def _monitor() -> TemporalResidualMonitor:
    return TemporalResidualMonitor(
        {
            'anomaly_magnitude_threshold': 0.02,
            'acceleration_cusum': {
                'instability_direction': 'negative',
                'signal': 'observed_velocity',
                'reference_value': 0.5,
                'decision_threshold': 2.0,
                'smoothing_span': 2,
                'persistence_window': 2,
                'persistence_threshold': 0.25,
            },
            'regime': {'smoothing_span': 2},
        },
    )


def test_regime_baseline_excludes_monitoring_dates_even_when_windows_overlap() -> None:
    rng = np.random.default_rng(1)
    observed = rng.normal(0.0, 0.1, 30)
    observed[20:] -= np.square(np.arange(1, 11, dtype=float))
    stack = observed[:, np.newaxis, np.newaxis]
    dates = tuple(f'2020-01-{day:02d}' for day in range(1, 31))

    overlapping = _monitor().analyze(
        stack, np.zeros_like(stack), dates, (0, 30), (15, 30)
    )
    prior_only = _monitor().analyze(
        stack, np.zeros_like(stack), dates, (0, 15), (15, 30)
    )

    assert overlapping.regime_calibration['last_index'] < 15
    assert (
        overlapping.regime_calibration['sigma']
        == prior_only.regime_calibration['sigma']
    )
    assert np.all(np.isnan(overlapping.regime_change_score[:15]))
    assert np.all(np.isnan(overlapping.regime_calibration_score[15:]))
    assert overlapping.regime_change_score[-1] > 0.7


def test_regime_score_requires_pre_monitoring_history() -> None:
    stack = np.arange(8, dtype=float)[:, np.newaxis, np.newaxis]
    dates = tuple(f'2020-01-{day:02d}' for day in range(1, 9))

    with pytest.raises(ValueError, match='precede monitoring'):
        _monitor().analyze(stack, np.zeros_like(stack), dates, (0, 8), (0, 8))


def test_frozen_calibration_is_written_once_and_rejects_changed_history(
    tmp_path,
) -> None:
    rng = np.random.default_rng(3)
    stack = rng.normal(0.0, 0.1, (20, 1, 1))
    dates = tuple(f'2020-01-{day:02d}' for day in range(1, 21))
    path = tmp_path / 'l3.json'
    baseline = (
        _monitor()
        .analyze(stack, np.zeros_like(stack), dates, (0, 20), (10, 20))
        .regime_calibration
    )

    payload = frozen_level3_calibration(
        path, baseline, dates, unit='mm/day', target_feature='velocity'
    )

    assert payload['last_date'] < dates[10]
    assert json.loads(path.read_text())['sigma'] == pytest.approx(baseline['sigma'])
    assert (
        frozen_level3_calibration(
            path, baseline, dates, unit='mm/day', target_feature='velocity'
        )
        == payload
    )
    changed = stack.copy()
    changed[3] += 1.0
    changed_baseline = (
        _monitor()
        .analyze(changed, np.zeros_like(changed), dates, (0, 20), (10, 20))
        .regime_calibration
    )
    with pytest.raises(ValueError, match='history changed'):
        frozen_level3_calibration(
            path, changed_baseline, dates, unit='mm/day', target_feature='velocity'
        )


def test_metadata_states_heuristic_and_lists_monitoring_levels() -> None:
    score = np.array([0.0, 0.5, 0.5])
    result = compute_level3(
        score,
        np.full(3, np.nan),
        _level2(np.zeros(3, dtype=bool)),
        (1, 3),
        PARAMETERS,
    )
    dates = ('2020-01-01', '2020-01-13', '2020-01-25')

    metadata = level3_metadata(result, dates)

    assert 'not a probability' in metadata['interpretation']
    assert metadata['first_watch_date'] == dates[2]
    assert metadata['level_counts'] == {'normal': 1, 'watch': 1, 'alert': 0}
    assert [row['date'] for row in metadata['series']] == list(dates[1:])
    assert metadata['series'][1]['l3_level'] == 'watch'
