import pandas as pd
import pytest

from subsystems.dpr.preprocessing_pipelines.sentinel1 import Sentinel1Pipeline


def all_pairs(baselines):
    """Mimic Stack.sbas_pairs() without limits for {date: perpendicular baseline}."""
    dates = sorted(pd.to_datetime(list(baselines)))
    rows = [
        {
            'ref': a,
            'rep': b,
            'ref_baseline': baselines[a.strftime('%Y-%m-%d')],
            'rep_baseline': baselines[b.strftime('%Y-%m-%d')],
        }
        for i, a in enumerate(dates)
        for b in dates[i + 1 :]
    ]
    df = pd.DataFrame(rows)
    return df.assign(
        pair=[f'{a.date()} {b.date()}' for a, b in zip(df.ref, df.rep)],
        baseline=df.rep_baseline - df.ref_baseline,
        duration=(df.rep - df.ref).dt.days,
    )


def limited(pairs, days, meters):
    return pairs[(pairs.duration <= days) & (pairs.baseline.abs() <= meters)]


BASELINES = {
    '2026-04-03': 28.6,
    '2026-04-09': -100.9,
    '2026-05-03': -54.4,
    '2026-05-04': -10.5,
    '2026-06-02': 231.4,
    '2026-06-08': -73.7,
    '2026-07-09': -99.0,
}


def test_outlier_date_is_bridged_with_cheapest_pair():
    full = all_pairs(BASELINES)
    pairs, report = Sentinel1Pipeline._repair_network(
        list(BASELINES), limited(full, 36, 80), full, 80, 180, 400.0, 0.1
    )
    assert report['dropped_dates'] == []
    assert [b['pair'] for b in report['bridges']] == ['2026-05-04 2026-06-02']
    assert len(Sentinel1Pipeline._network_components(list(BASELINES), pairs)) == 1


def test_unbridgeable_date_is_dropped_within_limit():
    full = all_pairs(BASELINES)
    pairs, report = Sentinel1Pipeline._repair_network(
        list(BASELINES), limited(full, 36, 80), full, 80, 180, 150.0, 0.2
    )
    assert report['dropped_dates'] == ['2026-06-02']
    assert not pairs.pair.str.contains('2026-06-02').any()
    remaining = [d for d in BASELINES if d != '2026-06-02']
    assert len(Sentinel1Pipeline._network_components(remaining, pairs)) == 1


def test_too_many_dropped_dates_fail():
    full = all_pairs(BASELINES)
    with pytest.raises(RuntimeError, match='cannot be bridged'):
        Sentinel1Pipeline._repair_network(
            list(BASELINES), limited(full, 36, 80), full, 80, 180, 150.0, 0.1
        )
