"""Tests for bounded AMD gap filling and retrospective smoothing."""

from datetime import date

import numpy as np

from subsystems.dag.plugins.eda.amd_trends import fill_short_gaps, robust_lowess


def test_fill_short_gaps_uses_dates_and_rejects_long_gaps():
    dates = (date(2020, 6, 1), date(2020, 6, 6), date(2020, 6, 11),
             date(2020, 12, 31), date(2021, 1, 5))
    values = np.array([0, np.nan, 10, 20, 30], dtype=float)
    filled, flags, gaps = fill_short_gaps(values, dates, max_gap_days=15)
    np.testing.assert_allclose(filled[:3], [0, 5, 10])
    assert flags.tolist() == [False, True, False, False, False]
    assert gaps[1] == 10


def test_lowess_is_robust_to_an_isolated_spike_and_has_no_extrapolation():
    dates = tuple(date(2020, 6, day) for day in (1, 6, 11, 16, 21, 26))
    values = np.array([np.nan, 1, 2, 30, 4, np.nan], dtype=float)
    trend = robust_lowess(values, dates, window_days=20, robust_iterations=2)
    assert np.isnan(trend[0]) and np.isnan(trend[-1])
    assert trend[3] < 15
