#!/usr/bin/env python3
"""ALE demonstration on the simulated showcase site (engine A, B or both).

Generates the showcase site once (MAP-format residual table, TSF mask,
sectors, zones, ground truth; see ale_3levels.demo.write_showcase_site), runs
the selected engine(s), and summarises the alerts per zone. With both engines
it also writes the comparison report and the plots (maps, histories, GIF).

Usage (from the repository root):
    python3 subsystems/ale/scripts/run_demo.py                 # both engines
    python3 subsystems/ale/scripts/run_demo.py --engine A      # fixed gates only
    python3 subsystems/ale/scripts/run_demo.py --engine B      # three levels only
    python3 subsystems/ale/scripts/run_demo.py --new-site      # regenerate the site

Default ``--out``: ``$ALE_DATA_DIR/showcase``, where ``ALE_DATA_DIR`` defaults to
``~/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV``. Outputs:
``<out>/runs/{fixed_gates,three_level}``, ``<out>/zone_summary.md``, and with
both engines ``<out>/report`` and ``<out>/plots``. Each call replaces earlier
runs; the site is kept unless ``--new-site`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

ALE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ALE_ROOT.parents[1]
sys.path.insert(0, str(ALE_ROOT))

import numpy as np
import rasterio
from ale_3levels.compare import compare
from ale_3levels.demo import SHOWCASE_ZONES, write_showcase_site
from ale_3levels.outputs import read_alerts
from ale_3levels.plots import plot_run
from ale_3levels.run import ENGINE_ALIASES, ENGINES, run
from ale_3levels.spatial import aggregate_sectors

DEFAULT_DATA_DIR = Path('~/ownCloud/Projects/GAIA_TSF/tsf_experiments/ALE-DEV')
MARKER = 'ale_showcase.yaml'
#: --quick: a small, short site for a fast check (fewer, smaller zones).
QUICK = {'height': 40, 'width': 52, 'n_calibration': 100, 'n_monitoring': 40}

ZONE_NOTES = {
    'stable': 'slow uniform settlement, forecast by MAP',
    'creep': 'steady consolidation ~7.5 mm/30 d, forecast by MAP',
    'failure': 'dam wall accelerating, not forecast by MAP',
}


def zone_levels(run_dir: Path, zones: np.ndarray, engines) -> dict:
    """Per engine: monitoring dates, level labels, and each zone's level series.

    Zones are aggregated like sectors and like the comparison report's areas
    (the engine's sector rule: coherent pixels, ``min_pixels`` or
    ``min_fraction``), so the numbers agree with the report.
    """
    names = [zone for zone, code in SHOWCASE_ZONES.items() if code]
    areas = {zone: zones == SHOWCASE_ZONES[zone] for zone in names}
    levels = {}
    for engine in engines:
        arrays, metadata = read_alerts(run_dir / engine)
        monitoring = arrays['dates'] >= metadata['input']['monitoring_start']
        rule = metadata['sector_rule']
        aggregated = aggregate_sectors(
            arrays['severity'][monitoring],
            areas,
            rule['coherence'],
            rule['min_pixels'],
            rule['min_fraction'],
        )
        levels[engine] = {
            'dates': arrays['dates'][monitoring],
            'labels': metadata['levels'][1:5],
            'zones': {
                zone: aggregated.severity[:, position]
                for position, zone in enumerate(names)
            },
        }
    return levels


def zone_summary(levels: dict) -> str:
    """Per engine and zone: first flagged date per level and dates flagged."""
    lines = []
    n_dates = 0
    for engine, data in levels.items():
        dates, n_dates = data['dates'], len(data['dates'])
        for zone, series in data['zones'].items():
            for level, label in zip((1, 2, 3), data['labels'][1:]):
                flagged = series >= level
                first = str(dates[np.argmax(flagged)]) if flagged.any() else '-'
                lines.append(
                    f'| {engine} | {zone} | >= {label} | {first} | {int(flagged.sum())} |'
                )
    header = [
        '# ALE demo: alerts per zone',
        '',
        (
            f'{n_dates} monitoring dates. A zone holds a level on a date when enough '
            'coherent pixels are at that level or higher (the sector rule, as in '
            'the comparison report).'
        ),
        '',
        '| engine | zone | level | first flagged | dates flagged |',
        '|---|---|---|---|---|',
    ]
    notes = ['', 'Zones: ' + '; '.join(f'{k}: {v}' for k, v in ZONE_NOTES.items())]
    return '\n'.join(header + lines + notes) + '\n'


def zone_displacement(table: Path, zones: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mean observed cumulative LOS (mm) per zone and acquisition, from MAP's table."""
    with np.load(table, allow_pickle=False) as archive:
        rows, cols = archive['row'], archive['col']
        targets = archive['target_date']
        observed = archive['observed_value']
    dates, index = np.unique(targets, return_inverse=True)
    codes = zones[rows, cols]
    means = {}
    for zone, code in SHOWCASE_ZONES.items():
        if not code:
            continue
        keep = codes == code
        total = np.bincount(index[keep], observed[keep], minlength=dates.size)
        count = np.bincount(index[keep], minlength=dates.size)
        means[zone] = total / np.maximum(count, 1)
    return dates.astype('datetime64[D]'), means


def zone_timeline(
    path: Path, table: Path, zones: np.ndarray, levels: dict, scenario: dict
) -> Path:
    """Time plot per zone: mean displacement and each engine's zone level."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dates, displacement = zone_displacement(table, zones)
    names = list(displacement)
    figure, axes = plt.subplots(
        2 * len(names),
        1,
        figsize=(12, 3.0 * len(names) + 1),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1] * len(names)},
    )
    events = [
        (scenario['monitoring'][0], 'monitoring starts', 'black'),
        (scenario['failure_onset'], 'failure begins', 'red'),
        (scenario['atmospheric_artefact'], 'atmospheric artefact', 'grey'),
    ]
    styles = {
        'fixed_gates': ('A  fixed_gates', '#1f77b4'),
        'three_level': ('B  three_level', '#9467bd'),
    }
    for position, zone in enumerate(names):
        top, bottom = axes[2 * position], axes[2 * position + 1]
        top.plot(dates, displacement[zone], color='black', lw=1.2)
        top.set_ylabel('mean LOS (mm)')
        top.set_title(f'{zone}: {ZONE_NOTES[zone]}', loc='left', fontsize=10)
        for engine, data in levels.items():
            label, colour = styles[engine]
            # Extend by one 12-day step so the last date's level is drawn.
            days = data['dates'].astype('datetime64[D]')
            series = data['zones'][zone]
            bottom.step(
                np.append(days, days[-1] + np.timedelta64(12, 'D')),
                np.append(series, series[-1]),
                where='post',
                color=colour,
                lw=1.8,
                label=label + ' (' + ' / '.join(data['labels'][1:]) + ')',
            )
        bottom.set_ylim(-0.3, 3.4)
        bottom.set_yticks([0, 1, 2, 3])
        bottom.set_ylabel('zone level')
        bottom.grid(axis='y', alpha=0.3)
        bottom.legend(loc='upper left', fontsize=8)
        for axis in (top, bottom):
            for day, text, colour in events:
                axis.axvline(np.datetime64(day), color=colour, ls='--', lw=0.8)
        if position == 0:
            for day, text, colour in events:
                top.annotate(
                    text,
                    (np.datetime64(day), 1.0),
                    xycoords=('data', 'axes fraction'),
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=8,
                    color=colour,
                )
    figure.suptitle('ALE showcase: displacement and alert level per zone')
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    data_dir = Path(os.environ.get('ALE_DATA_DIR', DEFAULT_DATA_DIR)).expanduser()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=data_dir / 'showcase')
    parser.add_argument(
        '--engine',
        choices=('A', 'B', 'both', *ENGINES),
        default='both',
        help='A = fixed_gates, B = three_level, or both (default)',
    )
    parser.add_argument('--new-site', action='store_true', help='regenerate the site')
    parser.add_argument('--no-plots', action='store_true', help='skip plots and GIF')
    parser.add_argument('--quick', action='store_true', help='small site, fast check')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

    out = args.out.expanduser().resolve()
    if out == REPOSITORY or REPOSITORY in out.parents:
        parser.error(f'--out must be outside the repository: {out}')
    if out.exists() and any(out.iterdir()) and not (out / MARKER).is_file():
        parser.error(f'{out} is not empty and is not a showcase site.')
    if args.new_site and out.exists():
        shutil.rmtree(out)
    if not (out / MARKER).is_file():
        write_showcase_site(out, **(QUICK if args.quick else {}))
        print(f'showcase site written: {out}')
    else:
        print(f'showcase site: {out}')
    config = out / MARKER
    scenario = json.loads((out / 'scenario.json').read_text())

    # Fresh runs every time (ALE outputs are written once per directory).
    for stale in ('runs', 'report', 'plots', 'zone_summary.md'):
        target = out / stale
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    engine = ENGINE_ALIASES.get(args.engine, args.engine)
    engines = ENGINES if engine == 'both' else (engine,)
    run(engine, out / 'runs', config_path=config)

    with rasterio.open(out / 'static' / 'demo_zones.tif') as raster:
        zones = raster.read(1)
    levels = zone_levels(out / 'runs', zones, engines)
    text = zone_summary(levels)
    (out / 'zone_summary.md').write_text(text, encoding='utf-8')
    print(
        f'\nmonitoring {scenario["monitoring"][0]} .. {scenario["monitoring"][1]}; '
        f'failure begins {scenario["failure_onset"]}; '
        f'atmospheric artefact {scenario["atmospheric_artefact"]}\n'
    )
    print(text)
    if not args.no_plots:
        timeline = zone_timeline(
            out / 'plots' / 'zone_timeline.png',
            out / 'results' / 'forecasts' / 'slope_trf' / 'residuals.npz',
            zones,
            levels,
            scenario,
        )
        print(f'time plot: {timeline}')
    if engine == 'both':
        report = compare(
            out / 'runs' / 'fixed_gates',
            out / 'runs' / 'three_level',
            out / 'report',
            truth=out / 'static' / 'roi_failure_mask.tif',
            title='ALE showcase site (simulated)',
        )
        print(f'comparison report: {report["report"]}')
        if not args.no_plots:
            plot_run(out / 'runs', out / 'plots', maps=True, animation=True)
            print(f'plots and animation: {out / "plots"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
