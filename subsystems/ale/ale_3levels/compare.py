"""``compare``: engine A vs engine B on the same input, with ground truth.

Reads two engine output directories (common schema) and, optionally, a
ground-truth failure mask. **The ground-truth mask is read only here**; no
engine, calibration or configuration default uses it.

Evaluated on monitoring dates only (``input.monitoring_start``). Severity is
the ordinal mapping (WHITE/NONE=0 ... RED/L3=3), for comparison only.

Areas for the ground-truth evaluation:

* ``failure``: the truth mask within the TSF mask;
* ``control``: an equally sized compact stable patch: the TSF pixels
  nearest to the TSF pixel farthest from the failure area, excluding a buffer
  around it (deterministic);

each aggregated with the same sector rule the engines' sector summaries use.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ale_3levels.outputs import read_alerts
from ale_3levels.spatial import aggregate_sectors

LEVEL_NAMES = {
    'fixed_gates': ('WHITE', 'GREEN', 'YELLOW', 'RED'),
    'three_level': ('NONE', 'L1', 'L2', 'L3'),
}
CONTROL_BUFFER_PIXELS = 5


def _days(dates) -> np.ndarray:
    return np.asarray(dates).astype('<U10').astype('datetime64[D]')


def control_area(
    mask: np.ndarray, failure: np.ndarray, buffer: int = CONTROL_BUFFER_PIXELS
) -> np.ndarray:
    """Compact stable patch, same size as ``failure``, far from it."""
    rows, cols = np.nonzero(failure)
    if rows.size == 0:
        raise ValueError('The failure area is empty within the TSF mask.')
    grid_rows, grid_cols = np.indices(mask.shape)
    distance = np.full(mask.shape, np.inf)
    for row, col in zip(rows, cols):
        distance = np.minimum(distance, np.hypot(grid_rows - row, grid_cols - col))
    candidates = mask & ~failure & (distance > buffer)
    if np.count_nonzero(candidates) < rows.size:
        raise ValueError('Not enough stable pixels for an equally sized control area.')
    seed = np.unravel_index(np.argmax(np.where(candidates, distance, -1)), mask.shape)
    to_seed = np.hypot(grid_rows - seed[0], grid_cols - seed[1])
    order = np.lexsort((grid_cols.ravel(), grid_rows.ravel(), to_seed.ravel()))
    chosen = [index for index in order if candidates.ravel()[index]][: rows.size]
    control = np.zeros(mask.size, dtype=bool)
    control[chosen] = True
    return control.reshape(mask.shape)


def _load(engine_dir: Path) -> dict[str, Any]:
    arrays, metadata = read_alerts(engine_dir)
    sectors = list(csv.DictReader((engine_dir / 'sectors.csv').open(encoding='utf-8')))
    return {'arrays': arrays, 'meta': metadata, 'sectors': sectors, 'dir': engine_dir}


def _check_same_input(a: dict[str, Any], b: dict[str, Any]) -> None:
    ia, ib = a['meta']['input'], b['meta']['input']
    for key in ('table_sha256', 'model_version', 'lead', 'monitoring_start'):
        if ia.get(key) != ib.get(key):
            raise ValueError(f'Engines were run on different inputs ({key}).')
    if not np.array_equal(a['arrays']['dates'], b['arrays']['dates']):
        raise ValueError('Engine outputs have different acquisition dates.')
    if not np.array_equal(a['arrays']['mask'], b['arrays']['mask']):
        raise ValueError('Engine outputs have different TSF masks.')


def _first(dates: np.ndarray, flags: np.ndarray) -> str | None:
    index = np.flatnonzero(flags)
    return None if index.size == 0 else str(dates[index[0]])


def _lead(first_a: str | None, first_b: str | None) -> int | None:
    if first_a is None or first_b is None:
        return None
    return int((_days([first_b])[0] - _days([first_a])[0]).astype(int))


def compare(
    a_dir: Path,
    b_dir: Path,
    out_dir: Path,
    *,
    truth: Path | None = None,
    title: str = 'ALE engine comparison',
) -> dict[str, Path]:
    """Write the comparison report (Markdown), its CSV tables and a JSON summary."""
    a, b = _load(Path(a_dir)), _load(Path(b_dir))
    if a['meta']['engine'] != 'fixed_gates' or b['meta']['engine'] != 'three_level':
        raise ValueError(
            '--a must be a fixed_gates output and --b a three_level output.'
        )
    _check_same_input(a, b)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = a['arrays']['dates']
    mask = a['arrays']['mask']
    start = a['meta']['input']['monitoring_start']
    monitoring = dates >= start
    sev_a = a['arrays']['severity'][monitoring]
    sev_b = b['arrays']['severity'][monitoring]
    mdates = dates[monitoring]
    tables: dict[str, Path] = {}

    # Sector timeline.
    timeline = []
    sector_a = {(r['date'], r['sector_id']): r for r in a['sectors']}
    sector_b = {(r['date'], r['sector_id']): r for r in b['sectors']}
    sector_ids = sorted({r['sector_id'] for r in a['sectors']})
    for day in mdates:
        for sector in sector_ids:
            ra, rb = sector_a[(str(day), sector)], sector_b[(str(day), sector)]
            timeline.append(
                [
                    str(day),
                    sector,
                    ra['level'],
                    int(ra['severity']),
                    rb['level'],
                    int(rb['severity']),
                ]
            )
    tables['timeline'] = _csv(
        out_dir / 'sector_timeline.csv',
        ['date', 'sector_id', 'a_level', 'a_severity', 'b_level', 'b_severity'],
        timeline,
    )

    # Confusion matrices (rows A, columns B; -1 = NO_DATA).
    codes = (-1, 0, 1, 2, 3)
    pixel_cells = np.broadcast_to(mask, sev_a.shape)
    pixel_matrix = _confusion(sev_a[pixel_cells], sev_b[pixel_cells], codes)
    sector_matrix = _confusion(
        np.array([row[3] for row in timeline]),
        np.array([row[5] for row in timeline]),
        codes,
    )
    tables['confusion_pixel'] = _matrix_csv(
        out_dir / 'confusion_pixel.csv', pixel_matrix, codes
    )
    tables['confusion_sector'] = _matrix_csv(
        out_dir / 'confusion_sector.csv', sector_matrix, codes
    )

    # Alert counts per acquisition.
    counts = []
    for index, day in enumerate(mdates):
        row = [str(day)]
        for sev in (sev_a, sev_b):
            values = sev[index][mask]
            row += [int(np.count_nonzero(values >= level)) for level in (1, 2, 3)]
        counts.append(row)
    tables['alert_counts'] = _csv(
        out_dir / 'alert_counts.csv',
        ['date', 'a_ge1', 'a_ge2', 'a_ge3', 'b_ge1', 'b_ge2', 'b_ge3'],
        counts,
    )

    # First alert per sector and level, lead-time difference (B minus A, days).
    first_rows = []
    for sector in sector_ids:
        sa = np.array([row[3] for row in timeline if row[1] == sector])
        sb = np.array([row[5] for row in timeline if row[1] == sector])
        for level in (1, 2, 3):
            fa, fb = _first(mdates, sa >= level), _first(mdates, sb >= level)
            first_rows.append([sector, level, fa, fb, _lead(fa, fb)])
    tables['first_alerts'] = _csv(
        out_dir / 'first_alerts.csv',
        ['sector_id', 'severity_at_least', 'a_first', 'b_first', 'b_minus_a_days'],
        first_rows,
    )

    # Flip rates between consecutive acquisitions (pixels valid at both).
    flips = {
        'fixed_gates': _flip_rate(
            a['arrays']['level'][monitoring],
            mask,
            a['meta']['levels'],
            'WHITE',
            'GREEN',
        ),
        'three_level': _flip_rate(
            b['arrays']['level'][monitoring], mask, b['meta']['levels'], 'NONE', 'L1'
        ),
    }
    tables['flips'] = _csv(
        out_dir / 'flip_rates.csv',
        ['engine', 'pair', 'flips', 'transitions', 'rate'],
        [
            [engine, value['pair'], value['flips'], value['transitions'], value['rate']]
            for engine, value in flips.items()
        ],
    )

    truth_summary = None
    if truth is not None:
        truth_summary, truth_rows = _ground_truth(a, b, truth, mask, monitoring, mdates)
        tables['ground_truth'] = _csv(
            out_dir / 'ground_truth.csv',
            [
                'engine',
                'area',
                'severity_at_least',
                'first_date',
                'dates_flagged',
                'monitoring_dates',
                'pixel_date_rate',
            ],
            truth_rows,
        )

    summary = {
        'title': title,
        'engines': {
            name: {
                'engine_version': data['meta']['engine_version'],
                'config_hash': data['meta']['config_hash'],
                'dir': str(data['dir']),
            }
            for name, data in (('fixed_gates', a), ('three_level', b))
        },
        'input': a['meta']['input'],
        'engine_b_calibration': _b_calibration(b['meta']),
        'monitoring_dates': int(monitoring.sum()),
        'pixel_confusion': pixel_matrix.tolist(),
        'sector_confusion': sector_matrix.tolist(),
        'first_alerts': first_rows,
        'flip_rates': flips,
        'ground_truth': truth_summary,
        'sector_rule': a['meta']['sector_rule'],
    }
    tables['summary'] = out_dir / 'report.json'
    tables['summary'].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    tables['report'] = out_dir / 'report.md'
    tables['report'].write_text(
        _markdown(summary, timeline, pixel_matrix, sector_matrix, counts, codes),
        encoding='utf-8',
    )
    return tables


def _b_calibration(meta: dict[str, Any]) -> dict[str, Any]:
    attrs = meta.get('engine_attrs', {})
    window = attrs.get('calibration_window') or [0, 0]
    return {
        'zscore_formula': attrs.get('zscore_formula'),
        'calibration_dates': int(window[1] - window[0]),
        'n_calibration': attrs.get('n_calibration'),
        'l1_sigma': attrs.get('l1_sigma'),
        'calibration_population': attrs.get(
            'calibration_population', 'prototype (all pre-monitoring dates)'
        ),
    }


def _confusion(a: np.ndarray, b: np.ndarray, codes) -> np.ndarray:
    matrix = np.zeros((len(codes), len(codes)), dtype=np.int64)
    for i, ca in enumerate(codes):
        for j, cb in enumerate(codes):
            matrix[i, j] = int(np.count_nonzero((a == ca) & (b == cb)))
    return matrix


def _flip_rate(levels, mask, labels, first, second) -> dict[str, Any]:
    labels = list(labels)
    one, two = labels.index(first), labels.index(second)
    previous, current = levels[:-1][:, mask], levels[1:][:, mask]
    valid = (previous > 0) & (current > 0)  # 0 = NO_DATA code
    flips = valid & (
        ((previous == one) & (current == two)) | ((previous == two) & (current == one))
    )
    transitions = int(valid.sum())
    return {
        'pair': f'{first}<->{second}',
        'flips': int(flips.sum()),
        'transitions': transitions,
        'rate': None if transitions == 0 else float(flips.sum() / transitions),
    }


def _ground_truth(a, b, truth: Path, mask, monitoring, mdates):
    from ale_3levels.contract import read_mask

    failure = read_mask(truth)
    if failure.shape != mask.shape:
        raise ValueError(
            f'Truth mask shape {failure.shape} differs from the grid {mask.shape}.'
        )
    failure &= mask
    control = control_area(mask, failure)
    rule = a['meta']['sector_rule']
    areas = {'failure': failure, 'control': control}
    rows, summary = (
        [],
        {
            'failure_pixels': int(failure.sum()),
            'control_pixels': int(control.sum()),
            'control_definition': (
                f'{int(control.sum())} TSF pixels nearest to the TSF pixel farthest from '
                f'the failure area, excluding a {CONTROL_BUFFER_PIXELS}-pixel buffer'
            ),
            'engines': {},
        },
    )
    for name, data in (('fixed_gates', a), ('three_level', b)):
        severity = data['arrays']['severity'][monitoring]
        aggregated = aggregate_sectors(
            severity, areas, rule['coherence'], rule['min_pixels'], rule['min_fraction']
        )
        engine = {}
        for position, area in enumerate(('failure', 'control')):
            cells = areas[area]
            for level in (1, 2, 3):
                flagged = aggregated.severity[:, position] >= level
                pixel_values = severity[:, cells]
                valid = pixel_values >= 0
                rate = (
                    float(np.count_nonzero(pixel_values >= level) / valid.sum())
                    if valid.any()
                    else None
                )
                first = _first(mdates, flagged)
                rows.append(
                    [
                        name,
                        area,
                        level,
                        first,
                        int(flagged.sum()),
                        int(flagged.size),
                        rate,
                    ]
                )
                engine[f'{area}_ge{level}'] = {
                    'first_date': first,
                    'dates_flagged': int(flagged.sum()),
                    'pixel_date_rate': rate,
                }
        engine['first_failure_alert'] = next(
            (
                {'date': engine[f'failure_ge{level}']['first_date'], 'severity': level}
                for level in (3, 2, 1)
                if engine[f'failure_ge{level}']['first_date'] is not None
            ),
            None,
        )
        summary['engines'][name] = engine
    summary['first_to_flag_failure'] = _first_flagger(summary['engines'])
    return summary, rows


def _first_flagger(engines: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for level in (1, 2, 3):
        firsts = {
            name: values[f'failure_ge{level}']['first_date']
            for name, values in engines.items()
        }
        known = {name: day for name, day in firsts.items() if day is not None}
        if not known:
            result[f'severity_ge{level}'] = None
            continue
        earliest = min(known.values())
        result[f'severity_ge{level}'] = {
            'engines': sorted(name for name, day in known.items() if day == earliest),
            'date': earliest,
            'all': firsts,
        }
    return result


def _csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(
            ['' if value is None else value for value in row] for row in rows
        )
    return path


def _matrix_csv(path: Path, matrix: np.ndarray, codes) -> Path:
    header = ['a_severity \\ b_severity', *[str(code) for code in codes]]
    rows = [[str(code), *matrix[i].tolist()] for i, code in enumerate(codes)]
    return _csv(path, header, rows)


def _md_table(header: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value)

    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]
    return '\n'.join(lines)


def _markdown(summary, timeline, pixel_matrix, sector_matrix, counts, codes) -> str:
    names = {-1: 'NO_DATA', 0: '0', 1: '1', 2: '2', 3: '3'}
    a_names = dict(zip(range(4), LEVEL_NAMES['fixed_gates'])) | {-1: 'NO_DATA'}
    b_names = dict(zip(range(4), LEVEL_NAMES['three_level'])) | {-1: 'NO_DATA'}
    inp = summary['input']
    cal = summary['engine_b_calibration']
    out = [f'# {summary["title"]}', '']
    out += [
        '## Run identity',
        '',
        _md_table(
            ['item', 'value'],
            [
                ['site', inp.get('site_id')],
                ['MAP model version', inp.get('model_version')],
                ['TSF version', inp.get('tsf_version')],
                ['MAP table SHA-256', (inp.get('table_sha256') or '')[:16]],
                [
                    'monitoring start / dates',
                    f'{inp.get("monitoring_start")} / {summary["monitoring_dates"]}',
                ],
                [
                    'contract calibration-split dates',
                    inp.get('calibration_split_dates'),
                ],
                [
                    'engine A version / config',
                    f'{summary["engines"]["fixed_gates"]["engine_version"]} / {summary["engines"]["fixed_gates"]["config_hash"][:12]}',
                ],
                [
                    'engine B version / config',
                    f'{summary["engines"]["three_level"]["engine_version"]} / {summary["engines"]["three_level"]["config_hash"][:12]}',
                ],
                ['engine B z-score formula', cal['zscore_formula']],
                ['engine B calibration dates', cal['calibration_dates']],
                ['engine B calibration population', cal['calibration_population']],
                ['engine B L1 sigma (mm/day)', cal['l1_sigma']],
            ],
        ),
        '',
        (
            'Severity is an ordinal mapping for comparison only (WHITE/NONE=0, GREEN/L1=1, '
            'YELLOW/L2=2, RED/L3=3, NO_DATA=-1); the levels of the two engines do not mean '
            'the same thing.'
        ),
        '',
    ]
    truth = summary['ground_truth']
    if truth:
        out += ['## Ground truth: failure area vs stable control area', '']
        out += [
            (
                f'Failure area: {truth["failure_pixels"]} pixels; control: {truth["control_definition"]}. '
                'Areas are aggregated with the same sector rule as the sector summaries. '
                '"dates flagged" counts monitoring dates on which the area held the level; '
                'for the failure area these are hits (the rest misses), for the control area '
                'false alerts.'
            ),
            '',
        ]
        rows = []
        for engine, values in truth['engines'].items():
            for level in (1, 2, 3):
                f, c = values[f'failure_ge{level}'], values[f'control_ge{level}']
                rows.append(
                    [
                        engine,
                        f'>= {level}',
                        f['first_date'],
                        f'{f["dates_flagged"]}/{summary["monitoring_dates"]}',
                        summary['monitoring_dates'] - f['dates_flagged'],
                        c['first_date'],
                        c['dates_flagged'],
                        f['pixel_date_rate'],
                        c['pixel_date_rate'],
                    ]
                )
        out += [
            _md_table(
                [
                    'engine',
                    'severity',
                    'failure: first alert',
                    'hits (dates)',
                    'misses',
                    'control: first alert',
                    'false alerts (dates)',
                    'failure pixel-date rate',
                    'control pixel-date rate',
                ],
                rows,
            ),
            '',
        ]
        for level in (3, 2, 1):
            first = truth['first_to_flag_failure'][f'severity_ge{level}']
            if first:
                out.append(
                    f'- Severity >= {level}: first flagged on {first["date"]} by '
                    f'{" and ".join(first["engines"])} (A: {first["all"]["fixed_gates"] or "never"}, '
                    f'B: {first["all"]["three_level"] or "never"}).'
                )
        out.append('')
    out += ['## First alert per sector and lead time (B minus A, days)', '']
    out += [
        _md_table(
            ['sector', 'severity', 'A first', 'B first', 'B − A (days)'],
            [[r[0], f'>= {r[1]}', r[2], r[3], r[4]] for r in summary['first_alerts']],
        ),
        '',
    ]
    out += ['## Sector timeline (changes only)', '']
    changes, previous = [], {}
    for row in timeline:
        key = row[1]
        state = (row[2], row[4])
        if previous.get(key) != state:
            changes.append([row[0], row[1], row[2], row[4]])
            previous[key] = state
    out += [_md_table(['date', 'sector', 'A level', 'B level'], changes), '']
    out += ['Full timeline: `sector_timeline.csv`.', '']
    for label, matrix in (
        ('pixel-dates', pixel_matrix),
        ('sector-dates', sector_matrix),
    ):
        out += [f'## Severity confusion matrix ({label}; rows A, columns B)', '']
        header = ['A \\ B', *[f'{names[c]} ({b_names[c]})' for c in codes]]
        rows = [
            [f'{names[c]} ({a_names[c]})', *matrix[i].tolist()]
            for i, c in enumerate(codes)
        ]
        agree = int(np.trace(matrix[1:, 1:]))
        total = int(matrix[1:, 1:].sum())
        out += [
            _md_table(header, rows),
            '',
            f'Same severity: {agree}/{total} ({agree / total:.1%}) of the cells where both have data.'
            if total
            else '',
            '',
        ]
    out += [
        '## Alert counts per acquisition (pixels at severity >= 1 / >= 2 / >= 3)',
        '',
    ]
    step = max(1, len(counts) // 12)
    out += [
        _md_table(
            ['date', 'A ≥1', 'A ≥2', 'A ≥3', 'B ≥1', 'B ≥2', 'B ≥3'],
            counts[::step],
        ),
        '',
        f'Every {step}. acquisition shown; all in `alert_counts.csv`.',
        '',
    ]
    out += ['## Noise near the lowest boundary', '']
    out += [
        _md_table(
            ['engine', 'pair', 'flips', 'transitions', 'rate'],
            [
                [
                    engine,
                    value['pair'],
                    value['flips'],
                    value['transitions'],
                    value['rate'],
                ]
                for engine, value in summary['flip_rates'].items()
            ],
        ),
        '',
        (
            'Share of consecutive-acquisition pixel transitions (both valid) that flip between '
            'the two lowest levels.'
        ),
        '',
    ]
    return '\n'.join(out)
