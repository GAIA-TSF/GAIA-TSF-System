"""Unit tests for AMD gap metrics."""

from datetime import date

import pytest

from subsystems.dag.plugins.eda.amd_gaps import analyse_amd_gaps


def test_gap_metrics_distinguish_calendar_days_and_missing_acquisitions():
    dates = (date(2020, 12, 20), date(2020, 12, 30),
             date(2021, 1, 9), date(2021, 2, 8))
    report, intervals = analyse_amd_gaps(dates, {'point:a': [True, False, True, True]})
    result = report['series']['point:a']
    assert report['candidate_cadence_days']['median'] == 10
    assert result['valid_gap_days']['max'] == 30
    assert result['within_year_valid_gap_days']['max'] == 30
    assert result['missing_acquisitions_between_valid']['max'] == 1
    assert result['yearly']['2020']['valid_fraction'] == 0.5
    assert result['yearly']['2021']['valid_fraction'] == 1
    assert intervals[0]['gap_days'] == 20
    assert intervals[0]['missing_acquisitions'] == 1


def test_gap_analysis_rejects_invalid_axes():
    with pytest.raises(ValueError, match='strictly chronological'):
        analyse_amd_gaps((date(2020, 1, 1), date(2020, 1, 1)), {'a': [1, 1]})
    with pytest.raises(ValueError, match='does not match'):
        analyse_amd_gaps((date(2020, 1, 1),), {'a': []})
