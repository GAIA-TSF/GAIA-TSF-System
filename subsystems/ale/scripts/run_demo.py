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


def zone_summary(run_dir: Path, zones: np.ndarray, engines) -> str:
    """Per engine and zone: first flagged date per level and dates flagged.

    Zones are aggregated like sectors and like the comparison report's areas
    (the engine's sector rule: coherent pixels, ``min_pixels`` or
    ``min_fraction``), so the numbers agree with the report.
    """
    names = [zone for zone, code in SHOWCASE_ZONES.items() if code]
    areas = {zone: zones == SHOWCASE_ZONES[zone] for zone in names}
    lines = []
    n_dates = 0
    for engine in engines:
        arrays, metadata = read_alerts(run_dir / engine)
        monitoring = arrays['dates'] >= metadata['input']['monitoring_start']
        dates, n_dates = arrays['dates'][monitoring], int(monitoring.sum())
        rule = metadata['sector_rule']
        aggregated = aggregate_sectors(
            arrays['severity'][monitoring],
            areas,
            rule['coherence'],
            rule['min_pixels'],
            rule['min_fraction'],
        )
        labels = metadata['levels'][2:5]
        for position, zone in enumerate(names):
            for level, label in zip((1, 2, 3), labels):
                flagged = aggregated.severity[:, position] >= level
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
    text = zone_summary(out / 'runs', zones, engines)
    (out / 'zone_summary.md').write_text(text, encoding='utf-8')
    print(
        f'\nmonitoring {scenario["monitoring"][0]} .. {scenario["monitoring"][1]}; '
        f'failure begins {scenario["failure_onset"]}; '
        f'atmospheric artefact {scenario["atmospheric_artefact"]}\n'
    )
    print(text)
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
