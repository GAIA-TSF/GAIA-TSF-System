"""Engine B quality states and series export (moved from MAP test_contract_v1)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ale_3levels.engines.three_level.quality import (
    QualityParameters,
    acquisition_quality,
    run_quality_flags,
)
from ale_3levels.engines.three_level.series import build_records, write_series


def test_quality_states_follow_contract_order() -> None:
    mask = np.ones((2, 2), dtype=bool)
    observed = np.ones((5, 2, 2))
    predicted = np.ones((5, 2, 2))
    predicted[1] = np.nan  # warm_up
    observed[2] = np.nan  # missing_acquisition
    observed[3, 0, :] = np.nan
    observed[3, 1, 0] = np.nan  # 25 % coverage

    states, coverage = acquisition_quality(
        observed, predicted, mask, (1, 5), QualityParameters()
    )

    assert states.tolist() == [
        'not_monitored',
        'warm_up',
        'missing_acquisition',
        'insufficient_coverage',
        'ok',
    ]
    assert coverage[3] == pytest.approx(0.25)


def test_short_calibration_is_flagged_and_warned(caplog) -> None:
    flags = run_quality_flags(
        run_mode='retrospective',
        l2_bias_warning=False,
        calibration_dates={'l1': 14, 'l2': 14, 'l3': 30},
        calibrations_reused=[],
        parameters=QualityParameters(),
    )

    names = [flag['flag'] for flag in flags]
    assert names == ['retrospective_training', 'short_calibration']
    assert flags[1]['calibration_dates'] == {'l1': 14, 'l2': 14}
    assert 'short_calibration' in caplog.text


def test_series_export_uses_canonical_names_and_nulls(tmp_path) -> None:
    records = build_records(
        {
            'date': ['2020-01-01', '2020-01-13'],
            'observed_mean_los_velocity': np.array([np.nan, -0.5]),
            'l2_alarm': np.array([False, True]),
            'l3_level': np.array(['not_evaluated', 'watch']),
        },
        range(2),
    )
    json_path, csv_path = write_series(
        tmp_path / 'monitoring_series', records, rate_unit='mm/day', metadata={}
    )

    payload = json.loads(json_path.read_text())
    assert payload['contract_version'] == '1.0'
    assert payload['series'][0]['observed_mean_los_velocity'] is None
    assert payload['units']['observed_mean_los_velocity'] == 'mm/day'
    assert csv_path.read_text().splitlines()[0] == (
        'date,observed_mean_los_velocity,l2_alarm,l3_level'
    )
    with pytest.raises(KeyError, match='prediction_uncertainty'):
        build_records({'prediction_uncertainty': [1.0]}, range(1))
