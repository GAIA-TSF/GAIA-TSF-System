"""Gap metrics for irregular, cloud-filtered AMD observations."""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import numpy as np


def analyse_amd_gaps(dates, series):
    """Summarise candidate-date availability for named Boolean time series."""
    dates = tuple(date.fromisoformat(value) if isinstance(value, str) else value for value in dates)
    if not dates or any(current <= previous for previous, current in pairwise(dates)):
        raise ValueError('AMD gap dates must be nonempty and strictly chronological.')
    candidate_gaps = np.asarray([(b - a).days for a, b in pairwise(dates)], dtype=float)
    report = {
        'definition': {
            'candidate_acquisition': 'Sentinel-2 acquisition retained after inventory and date filtering',
            'valid_observation': 'at least one finite AMD value for the scope',
            'gap_days': 'calendar days between consecutive valid observations',
            'missing_acquisitions': 'candidate acquisitions between consecutive valid observations',
        },
        'period': {'start': dates[0].isoformat(), 'end': dates[-1].isoformat()},
        'candidate_acquisitions': len(dates),
        'candidate_cadence_days': _stats(candidate_gaps),
        'series': {},
    }
    intervals = []
    for scope, values in series.items():
        valid = np.asarray(values, dtype=bool)
        if valid.shape != (len(dates),):
            raise ValueError(f'AMD gap series {scope!r} does not match dates.')
        indices = np.flatnonzero(valid)
        gaps = np.asarray([(dates[b] - dates[a]).days for a, b in pairwise(indices)], dtype=float)
        within_year_gaps = np.asarray(
            [(dates[b] - dates[a]).days for a, b in pairwise(indices)
             if dates[a].year == dates[b].year], dtype=float
        )
        missing = np.asarray([b - a - 1 for a, b in pairwise(indices)], dtype=float)
        years = {}
        for year in sorted({value.year for value in dates}):
            selected = np.asarray([value.year == year for value in dates])
            total = int(selected.sum())
            count = int((valid & selected).sum())
            years[str(year)] = {
                'candidate_acquisitions': total,
                'valid_observations': count,
                'missing_observations': total - count,
                'valid_fraction': count / total,
            }
        report['series'][scope] = {
            'valid_observations': int(valid.sum()),
            'missing_observations': int((~valid).sum()),
            'valid_fraction': float(valid.mean()),
            'valid_gap_days': _stats(gaps),
            'within_year_valid_gap_days': _stats(within_year_gaps),
            'missing_acquisitions_between_valid': _stats(missing),
            'yearly': years,
        }
        for a, b in pairwise(indices):
            intervals.append({
                'scope': scope,
                'previous_valid_date': dates[a].isoformat(),
                'next_valid_date': dates[b].isoformat(),
                'gap_days': (dates[b] - dates[a]).days,
                'missing_acquisitions': b - a - 1,
            })
    return report, intervals


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {'count': 0, 'min': None, 'median': None, 'mean': None,
                'p90': None, 'max': None}
    return {
        'count': int(values.size), 'min': float(values.min()),
        'median': float(np.median(values)), 'mean': float(values.mean()),
        'p90': float(np.percentile(values, 90)), 'max': float(values.max()),
    }
