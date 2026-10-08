"""Lossless, atomic per-pair checkpoints on the radar unwrapping grid."""
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xarray as xr


class PairCheckpoints:
    def __init__(self, root, config):
        self.path = Path(root) / 'checkpoints' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.path.mkdir(parents=True)
        self.manifest = {
            'schema_version': 1, 'status': 'running', 'coordinate_system': 'radar y/x; not EPSG:4326',
            'pygmtsar_version': version('pygmtsar'),
            'configuration': {key: str(value) for key, value in config.items()},
            'pairs': [],
            'notes': ['Phase units are radians. Coherence is dimensionless.',
                      'Unwrapped phase is saved before spatial referencing or detrending.',
                      'SNAPHU labels are local to each pair; 0 is unassigned/masked.',
                      'Labels are diagnostic and do not establish consistency between components.',
                      'These files support phase-domain inspection, not an automatic full-pipeline resume.'],
        }
        self.save()

    def save(self):
        path = self.path / 'manifest.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.manifest, indent=2, allow_nan=False))
        tmp.replace(path)

    def network(self, pairs, candidates):
        pairs.to_csv(self.path / 'sbas_pairs.csv', index=False)
        (self.path / 'sbas_pairs.json').write_text(pairs.to_json(orient='records', date_format='iso', indent=2))
        self.manifest['network_candidates'] = candidates
        self.save()

    def write(self, stage, dataset):
        """Commit each pair independently; only completed files enter the manifest."""
        if stage not in ('wrapped', 'unwrapped'):
            raise ValueError('Unknown checkpoint stage')
        if 'pair' not in dataset.dims:
            raise ValueError('Checkpoint dataset must have a pair dimension')
        ids = [str(value) for value in dataset.pair.values]
        if len(set(ids)) != len(ids):
            raise ValueError('Checkpoint pair identifiers must be unique')
        entries = self.manifest['pairs']
        if not entries:
            self.manifest['pairs'] = entries = [{'index': i, 'pair': pair} for i, pair in enumerate(ids)]
        if [entry['pair'] for entry in entries] != ids:
            raise ValueError('Checkpoint pair order differs from the saved wrapped input')
        for i, entry in enumerate(entries):
            # Preserve a length-one pair axis, ref/rep dates and exact radar coordinates.
            data = dataset.isel(pair=slice(i, i + 1)).compute()
            directory = self.path / 'pairs' / f'{i:06d}'
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f'{stage}.nc'
            tmp = target.with_suffix('.nc.tmp')
            encoding = {name: {'zlib': True, 'complevel': 4} for name in data.data_vars}
            try:
                data.to_netcdf(tmp, engine='h5netcdf', encoding=encoding)
                tmp.replace(target)
            finally:
                if tmp.exists():
                    tmp.unlink()
            entry[stage] = str(target.relative_to(self.path))
            for name in ('wrapped_phase', 'unwrapped_phase'):
                if name in data:
                    entry[f'{name}_finite_pixels'] = int(np.isfinite(data[name]).sum())
            self.save()
        self.manifest[f'{stage}_complete'] = True
        self.save()

    @staticmethod
    def open_stage(path, stage):
        """Open a committed complete stack lazily; caller owns/ closes the dataset."""
        path = Path(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        if stage not in ('wrapped', 'unwrapped') or not manifest.get(f'{stage}_complete'):
            raise ValueError(f'Checkpoint stage is incomplete: {stage}')
        files = [path / entry[stage] for entry in manifest['pairs']]
        return xr.open_mfdataset(files, engine='h5netcdf', combine='nested', concat_dim='pair',
                                 join='exact', data_vars='all', coords='minimal', compat='equals')
