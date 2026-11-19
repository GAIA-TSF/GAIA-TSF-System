"""Capture golden outputs of the three-level alert prototype (plan step 2).

The prototype (``subsystems/map/monitoring`` at f11c5f2 + stash@{0}) is run
unmodified, as a subprocess, from a checkout of that state. Its input is built
from a MAP forecast-residual table (``residuals.npz`` + ``residuals.json``,
schema 2.0) exactly as the prototype expects it (``monitoring_input.npz``):

* lead 1 rows with ``target_date_status == 'acquired'``;
* rates over the forecast interval, harmonized to mm/day:
  ``observed = (observed_value - persistence_value) / dt * scale``,
  ``prediction = (predicted_value - persistence_value) / dt * scale``,
  ``residual = residual / dt * scale``, with ``dt = target - last_observation``
  in days and ``scale`` converting ``--source-unit`` to mm;
* prototype windows: calibration = every date before the monitoring start in
  the table sidecar, monitoring = the rest.

The legacy global z-score (Z-A, ``anomaly_detection.py`` at 950702d) is run on
the same residual stack. All numerical outputs are packed into
``golden_<name>.npz`` + ``.json`` with the inputs needed to replay them.

This script imports neither MAP nor ALE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import yaml

UNIT_TO_MM = {'mm': 1.0, 'm': 1000.0}
# 950702d config.yaml anomaly_detection (the committed Z-A configuration).
LEGACY_ZA_CONFIG = {
    'residual_threshold': 0.1,
    'zscore_threshold': 1.5,
    'direction': 'negative',
    'persistence': 3,
}


def build_inputs(
    table_path: Path,
    mask_path: Path,
    source_unit: str,
    crop: tuple[int, int, int, int] | None,
) -> dict[str, np.ndarray]:
    """Return the prototype's ``monitoring_input.npz`` arrays."""
    sidecar = json.loads(table_path.with_suffix('.json').read_text(encoding='utf-8'))
    scale = UNIT_TO_MM[source_unit]
    with np.load(table_path, allow_pickle=False) as archive:
        table = {name: archive[name] for name in archive.files}
    keep = (table['lead'] == 1) & (table['target_date_status'] == 'acquired')
    table = {name: values[keep] for name, values in table.items()}
    target = np.array([value[:10] for value in table['target_date']])
    last = table['last_observation_date']
    dt = np.array(
        [
            (date.fromisoformat(t) - date.fromisoformat(o)).days
            for t, o in zip(target, last)
        ],
        dtype=np.float64,
    )
    if np.any(dt <= 0):
        raise ValueError('Lead-1 rows must have a positive forecast interval.')
    persistence = table['persistence_value']
    columns = {
        'observed_stack': (table['observed_value'] - persistence) / dt * scale,
        'prediction_stack': (table['predicted_value'] - persistence) / dt * scale,
        'residual_stack': table['residual'] / dt * scale,
    }
    dates = np.unique(target)
    height, width = int(sidecar['grid']['height']), int(sidecar['grid']['width'])
    time_index = np.searchsorted(dates, target)
    stacks = {}
    for name, values in columns.items():
        stack = np.full((dates.size, height, width), np.nan, dtype=np.float64)
        stack[time_index, table['row'], table['col']] = values
        stacks[name] = stack
    origins = np.empty(dates.size, dtype=object)
    origins[time_index] = table['forecast_origin_date']
    with rasterio.open(mask_path) as raster:
        mask = raster.read(1) > 0
    transform = list(sidecar['grid']['transform'])
    if crop is not None:
        r0, r1, c0, c1 = crop
        stacks = {name: stack[:, r0:r1, c0:c1] for name, stack in stacks.items()}
        mask = mask[r0:r1, c0:c1]
        transform[2] += c0 * transform[0]
        transform[5] += r0 * transform[4]
    monitoring_start = str(sidecar['monitoring_window'][0])
    start = int(np.searchsorted(dates, monitoring_start))
    if not 0 < start < dates.size:
        raise ValueError('Monitoring start must split the acquisition axis.')
    return {
        **stacks,
        'uncertainty_stack': np.empty((0, 0, 0), dtype=np.float64),
        'has_uncertainty': np.array(False),
        'mask': mask,
        'fixed_support_mask': mask.copy(),
        'dates': dates.astype('<U10'),
        'target_dates': dates.astype('<U10'),
        'forecast_origin_dates': origins.astype('<U10'),
        'source_label_dates': dates.astype('<U10'),
        'target_feature': np.asarray(f'{sidecar["target_feature"]}_rate_mm_per_day'),
        'forecast_timing_schema_version': np.asarray('1.0'),
        'run_mode': np.asarray('operational'),
        'run_mode_reason': np.asarray(
            'training pixels from the configured stable polygon'
        ),
        'calibration_window': np.array([0, start], dtype=np.int64),
        'monitoring_window': np.array([start, dates.size], dtype=np.int64),
        'transform': np.asarray(transform[:6], dtype=np.float64),
        'crs': np.asarray(str(sidecar['grid']['crs'])),
        'nodata': np.asarray(np.nan),
    }


def write_prototype_config(prototype: Path, run_dir: Path) -> Path:
    """Prototype config.yaml with this run's root and plots disabled."""
    source = prototype / 'subsystems' / 'map' / 'config.yaml'
    config = yaml.safe_load(source.read_text(encoding='utf-8'))
    config['experiment_dir'] = str(run_dir)
    config['monitoring']['dashboard']['enabled'] = False
    config['monitoring']['animation']['enabled'] = False
    path = run_dir / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return path


LEGACY_RUNNER = """
import importlib.util, json, sys
from types import SimpleNamespace
import numpy as np
spec = importlib.util.spec_from_file_location('legacy_ad', sys.argv[1])
module = importlib.util.module_from_spec(spec); sys.modules['legacy_ad'] = module
spec.loader.exec_module(module)
with np.load(sys.argv[2], allow_pickle=False) as a:
    residual, mask = a['residual_stack'], a['mask']
    dates = tuple(str(v) for v in a['dates'])
    start, end = (int(v) for v in a['monitoring_window'])
detector = module.StatisticalAnomalyDetector(json.loads(sys.argv[3]))
result = detector.detect(SimpleNamespace(mask=mask, dates=dates), residual, start, end)
np.savez_compressed(sys.argv[4], score=result.score_stack, binary=result.binary_stack)
with open(sys.argv[4].replace('.npz', '.json'), 'w') as stream:
    json.dump(result.summary, stream, indent=2, sort_keys=True)
"""


def run_prototype(prototype: Path, run_dir: Path, legacy_file: Path) -> None:
    config = write_prototype_config(prototype, run_dir)
    env = {'PYTHONPATH': str(prototype), 'PATH': '/usr/bin:/bin', 'MPLBACKEND': 'Agg'}
    subprocess.run(
        [
            sys.executable,
            str(prototype / 'subsystems/map/run_monitoring.py'),
            '--config',
            str(config),
        ],
        check=True,
        env=env,
        cwd=prototype,
    )
    subprocess.run(
        [
            sys.executable,
            '-c',
            LEGACY_RUNNER,
            str(legacy_file),
            str(run_dir / 'results/inference/monitoring_input.npz'),
            json.dumps(LEGACY_ZA_CONFIG),
            str(run_dir / 'legacy_za.npz'),
        ],
        check=True,
        env=env,
        cwd=prototype,
    )


RASTER_PRODUCTS = (
    'anomaly_score',
    'anomaly_binary',
    'anomaly_l1_binary',
    'anomaly_coherent',
    'residual_z',
)
MONITORING_RASTERS = (
    'level2_maximum_instability_cusum',
    'level2_first_alarm_index',
    'maximum_coherence_acceleration_cusum',
)
JSON_FILES = {
    'monitoring_metadata.json': 'results/monitoring/monitoring_metadata.json',
    'monitoring_series.json': 'results/monitoring/monitoring_series.json',
    'l1_calibration.json': 'results/monitoring/l1_calibration.json',
    'l2_calibration.json': 'results/monitoring/l2_calibration.json',
    'l3_calibration.json': 'results/monitoring/l3_calibration.json',
    'anomaly_summary.json': 'results/anomalies/anomaly_summary.json',
    'spatial_coherence_summary.json': 'results/anomalies/spatial_coherence_summary.json',
    'legacy_za_summary.json': 'legacy_za.json',
}


def _read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as raster:
        return raster.read(1)


def pack(run_dir: Path, out_dir: Path, provenance: dict[str, object]) -> None:
    """Copy the prototype's numerical outputs into one self-contained directory.

    Raster products are float32 GeoTIFFs in the prototype, so they are stored
    as written. Series, calibrations and Z-A are float64 and stored exactly.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    results = run_dir / 'results'
    with np.load(
        results / 'inference' / 'monitoring_input.npz', allow_pickle=False
    ) as archive:
        inputs = {name: archive[name] for name in archive.files}
        dates = [str(value).replace('-', '') for value in archive['dates']]
    np.savez_compressed(out_dir / 'inputs.npz', **inputs)
    rasters = {
        name: np.stack(
            [_read_band(results / 'anomalies' / f'{name}_{day}.tif') for day in dates]
        )
        for name in RASTER_PRODUCTS
    }
    rasters.update(
        {
            name: _read_band(results / 'monitoring' / f'{name}.tif')
            for name in MONITORING_RASTERS
        }
    )
    np.savez_compressed(out_dir / 'rasters.npz', **rasters)
    with np.load(run_dir / 'legacy_za.npz', allow_pickle=False) as archive:
        np.savez_compressed(
            out_dir / 'legacy_za.npz', **{n: archive[n] for n in archive.files}
        )
    for target, source in JSON_FILES.items():
        (out_dir / target).write_bytes((run_dir / source).read_bytes())
    (out_dir / 'config.yaml').write_bytes((run_dir / 'config.yaml').read_bytes())
    manifest = {
        **provenance,
        'input_sha256': hashlib.sha256(
            (out_dir / 'inputs.npz').read_bytes()
        ).hexdigest(),
        'legacy_za_config': LEGACY_ZA_CONFIG,
        'files': sorted(path.name for path in out_dir.iterdir()),
        'note': (
            'Absolute paths inside monitoring_metadata.json refer to the capture '
            'run directory and are not part of the comparison.'
        ),
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )


def _git_head(path: Path) -> str:
    return subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--out', type=Path, required=True, help='golden output directory'
    )
    parser.add_argument('--prototype-label', default='f11c5f2 + stash@{0}')
    parser.add_argument(
        '--residuals', type=Path, required=True, help='MAP residuals.npz'
    )
    parser.add_argument('--mask', type=Path, required=True, help='static/tsf_mask.tif')
    parser.add_argument('--source-unit', choices=sorted(UNIT_TO_MM), required=True)
    parser.add_argument(
        '--prototype', type=Path, required=True, help='checkout of the prototype'
    )
    parser.add_argument('--legacy-anomaly-detection', type=Path, required=True)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('R0', 'R1', 'C0', 'C1'))
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    inputs = build_inputs(
        args.residuals,
        args.mask,
        args.source_unit,
        tuple(args.crop) if args.crop else None,
    )
    if (args.run_dir / 'results').exists():
        # The prototype's calibration artifacts are create-once; never reuse them.
        raise FileExistsError(f'Run directory is not empty: {args.run_dir}')
    path = args.run_dir / 'results' / 'inference' / 'monitoring_input.npz'
    path.parent.mkdir(parents=True)
    # Old MAP inference created this directory; the prototype writes into it.
    (args.run_dir / 'results' / 'residuals').mkdir()
    np.savez_compressed(path, **inputs)
    run_prototype(
        args.prototype.resolve(), args.run_dir.resolve(), args.legacy_anomaly_detection
    )
    sidecar = json.loads(
        args.residuals.with_suffix('.json').read_text(encoding='utf-8')
    )
    pack(
        args.run_dir,
        args.out,
        {
            'description': 'Golden outputs of the three-level alert prototype (ALE plan step 2).',
            'prototype': args.prototype_label,
            'prototype_head': _git_head(args.prototype),
            'legacy_za_source': '950702d:subsystems/map/monitoring/anomaly_detection.py',
            'map_table': str(args.residuals.name),
            'map_table_sha256': hashlib.sha256(args.residuals.read_bytes()).hexdigest(),
            'map_model_version': sidecar.get('model_version'),
            'map_target_feature': sidecar.get('target_feature'),
            'source_unit': args.source_unit,
            'crop_rows_cols': args.crop,
            'lead': 1,
            'calibration_window_rule': 'all dates before the table monitoring_window start',
        },
    )
    print(f'golden written: {args.out}')


if __name__ == '__main__':
    main()
