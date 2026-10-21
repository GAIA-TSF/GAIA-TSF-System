"""Diagnose SBAS connectivity from saved aligned PRMs or an exported baseline CSV.

Example (inside the processing container):
  python -m subsystems.dpr.scripts.diagnose_sbas_network \
    --workdir /data/gaia_tsf/SITE/processing/sentinel1/pre_failure \
    --output-dir /data/gaia_tsf/SITE/diagnostics/network_review

Use --baseline-csv baselines.csv to repeat analysis after workdir cleanup.
No source data, exclusions or processing configuration are modified.
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path
import re


def read_baselines(workdir, reference_date=None, subswath=None):
    from pygmtsar import PRM

    files = {}
    for path in Path(workdir).glob('*.PRM'):
        match = re.fullmatch(r'(\d{4}-\d{2}-\d{2})_F(\d+)\.PRM', path.name)
        if match:
            files.setdefault(match[2], {})[match[1]] = path
    if subswath is None:
        if len(files) != 1:
            raise ValueError(f'Select --subswath from {sorted(files)}; expected one aligned PRM group.')
        subswath = next(iter(files))
    selected = files.get(str(subswath), {})
    if not selected:
        raise ValueError('No matching aligned YYYY-MM-DD_F*.PRM files.')
    reference_date = reference_date or min(selected)
    if reference_date not in selected:
        raise ValueError('Reference date is absent from the selected PRMs.')
    ref = PRM.from_file(str(selected[reference_date]))
    rows = []
    for day, path in sorted(selected.items()):
        baseline = float(ref.SAT_baseline(PRM.from_file(str(path))).get('B_perpendicular'))
        rows.append({'date': day, 'baseline_m': baseline})
    return rows, {'workdir': str(Path(workdir).resolve()), 'reference_date': reference_date,
                  'subswath': subswath}


def analyze(rows, days, meters):
    """Match PyGMTSAR sbas_pairs: integer-day limit and abs(Bdiff) < meters+1."""
    import math
    rows = sorted(rows, key=lambda r: r['date'])
    dates = [date.fromisoformat(r['date']) for r in rows]
    if len(set(dates)) != len(dates) or len(dates) < 2:
        raise ValueError('At least two unique acquisition dates are required.')
    if days <= 0 or meters <= 0 or not math.isfinite(meters):
        raise ValueError('Thresholds must be finite and positive.')
    if not all(math.isfinite(float(r['baseline_m'])) for r in rows):
        raise ValueError('Baselines must be finite.')
    adjacency = {r['date']: set() for r in rows}
    pairs, rejected = [], []
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            duration = (dates[j] - dates[i]).days
            delta = abs(float(b['baseline_m']) - float(a['baseline_m']))
            edge = {'ref': a['date'], 'rep': b['date'], 'days': duration, 'baseline_m': delta}
            if duration <= days and delta < meters + 1:
                pairs.append(edge)
                adjacency[a['date']].add(b['date'])
                adjacency[b['date']].add(a['date'])
            else:
                rejected.append(edge)
    groups, seen = [], set()
    for node in adjacency:
        if node in seen:
            continue
        todo, group = [node], []
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            group.append(current)
            todo.extend(adjacency[current] - seen)
        groups.append(sorted(group))
    groups.sort(key=lambda g: (-len(g), g[0]))
    membership = {d: i for i, group in enumerate(groups) for d in group}
    bridges = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            candidates = [e for e in rejected if {membership[e['ref']], membership[e['rep']]} == {i, j}]
            within_time = [e for e in candidates if e['days'] <= days]
            within_space = [e for e in candidates if e['baseline_m'] < meters + 1]
            bridges.append({'components': [i, j],
                            'nearest_spatial_within_time_limit': min(within_time, key=lambda e: e['baseline_m'], default=None),
                            'nearest_temporal_within_space_limit': min(within_space, key=lambda e: e['days'], default=None)})
    return {'thresholds': {'days': days, 'meters': meters, 'rule': 'days <= limit; abs(baseline difference) < meters + 1 (PyGMTSAR)'},
            'acquisitions': [dict(r, component=membership[r['date']], degree=len(adjacency[r['date']])) for r in rows],
            'components': groups, 'component_count': len(groups), 'pair_count': len(pairs),
            'pairs': pairs, 'isolated_dates': [g[0] for g in groups if len(g) == 1],
            'bridge_options': bridges}


def plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['acquisitions']
    lookup = {r['date']: r for r in rows}
    fig, ax = plt.subplots(figsize=(13, 6))
    for pair in report['pairs']:
        ax.plot([date.fromisoformat(pair[k]) for k in ('ref', 'rep')],
                [lookup[pair[k]]['baseline_m'] for k in ('ref', 'rep')], color='gray', alpha=.15, linewidth=.5)
    for index, group in enumerate(report['components']):
        ax.scatter([date.fromisoformat(d) for d in group], [lookup[d]['baseline_m'] for d in group],
                   s=18, label=f'Component {index}: {len(group)} dates')
    for d in report['isolated_dates']:
        ax.annotate(d, (date.fromisoformat(d), lookup[d]['baseline_m']), xytext=(4, 8), textcoords='offset points')
    ax.set(xlabel='Acquisition date (UTC)', ylabel='Perpendicular baseline relative to reference (m)',
           title=f"SBAS: {report['component_count']} components, {report['pair_count']} pairs")
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(output / 'sbas_network.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot([date.fromisoformat(r['date']) for r in rows], [r['degree'] for r in rows], '.-')
    ax.set(xlabel='Acquisition date (UTC)', ylabel='Number of incident pairs', title='Acquisition connectivity (zero = isolated)')
    ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(output / 'acquisition_degree.png', dpi=180); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--workdir', type=Path)
    source.add_argument('--baseline-csv', type=Path, help='CSV columns: date, baseline_m')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-date', help='Baseline calculation reference; defaults to earliest PRM date')
    parser.add_argument('--subswath', help='PRM suffix, e.g. 2 for F2')
    parser.add_argument('--days', type=int, default=120)
    parser.add_argument('--meters', type=float, default=150)
    parser.add_argument('--exclude-dates', nargs='*', default=[], help='Hypothetical exclusions only; source/config unchanged')
    args = parser.parse_args(argv)
    if args.workdir:
        rows, provenance = read_baselines(args.workdir, args.reference_date, args.subswath)
    else:
        with args.baseline_csv.open() as f:
            rows = [{'date': r['date'], 'baseline_m': float(r['baseline_m'])} for r in csv.DictReader(f)]
        provenance = {'baseline_csv': str(args.baseline_csv.resolve())}
    report = analyze(rows, args.days, args.meters)
    report['provenance'] = provenance
    report['notes'] = ['Connectivity is not phase-quality validation. Isolated acquisitions are not necessarily corrupt.',
                       'Bridge options are diagnostic candidates, not validated interferograms. No automatic exclusions.',
                       'Use the original alignment reference for the closest reproduction of pipeline baselines.']
    if args.exclude_dates:
        unknown = set(args.exclude_dates) - {r['date'] for r in rows}
        if unknown:
            raise ValueError(f'Unknown excluded dates: {sorted(unknown)}')
        report['exclusion_trial'] = analyze([r for r in rows if r['date'] not in args.exclude_dates], args.days, args.meters)
        report['exclusion_trial']['excluded_dates'] = args.exclude_dates
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / 'network_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (output / 'baselines.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'baseline_m']); writer.writeheader(); writer.writerows(rows)
    plots(report, output)
    if 'exclusion_trial' in report:
        trial_dir = output / 'exclusion_trial'; trial_dir.mkdir(exist_ok=True)
        plots(report['exclusion_trial'], trial_dir)
    print(f"{len(rows)} acquisitions; {report['component_count']} components; isolated: {report['isolated_dates']}")
    print(f'Report: {output / "network_report.json"}')


if __name__ == '__main__':
    main()
