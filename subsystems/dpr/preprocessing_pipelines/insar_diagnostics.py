"""Small, persistent InSAR diagnostics; no changes to the scientific solution."""
import json
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class InSARDiagnostics:
    def __init__(self, output, config):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.path = Path(output) / 'diagnostics' / stamp
        self.path.mkdir(parents=True)
        self.summary = {
            'started_utc': stamp, 'status': 'running', 'stages': [],
            'configuration': {k: str(v) for k, v in config.items()},
            'notes': ['Quality statistics are not displacement accuracy estimates.',
                      'Dates are UTC; plots in radar coordinates are not geographic maps.'],
        }
        self.summary['versions'] = {}
        for package in ('pygmtsar', 'numpy', 'xarray', 'dask', 'matplotlib'):
            try:
                self.summary['versions'][package] = version(package)
            except PackageNotFoundError:
                self.summary['versions'][package] = 'unknown'
        (self.path / 'README.md').write_text(
            '# InSAR run diagnostics\n\n'
            'Start with summary.json for status, failed stage, configuration and library versions.\n'
            'Stage timings include work performed in that stage; lazy computations may execute later.\n'
            'CSV and JSON tables contain the same records. Missing values are JSON null.\n\n'
            '- sbas_network.png: date versus perpendicular baseline; edges are selected pairs.\n'
            '- network_candidates: threshold combinations and their graph component counts.\n'
            '- acquisition_degree: pair count at each date; single-edge dates lack redundancy.\n'
            '- slc_coverage: valid aligned input pixels by date.\n'
            '- wrapped_phase / unwrapped_phase: pair coverage and mean accepted coherence.\n'
            '- sample PNGs: first and least-covered pair, in radar coordinates; phase is radians.\n'
            '- reference_phase: combined reference pixel counts and phase offsets by pair.\n'
            '- displacement_by_date: whole-grid cumulative LOS statistics in mm.\n\n'
            'Means can change with valid pixel support. Phase sample colour scales are independent.\n'
            'These are diagnostics, not uncertainty estimates, component/closure validation,\n'
            'or complete intermediate phase checkpoints. A partial run has only completed reports.\n'
            'When retention is enabled, summary.json links to separate per-pair NetCDF checkpoints.\n'
        )
        self.save()

    def save(self):
        target = self.path / 'summary.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.summary, indent=2, allow_nan=False))
        tmp.replace(target)

    def table(self, name, frame):
        frame.to_csv(self.path / f'{name}.csv', index=False)
        # pandas emits null for missing/nonfinite values: standards-compliant JSON.
        (self.path / f'{name}.json').write_text(frame.to_json(orient='records', date_format='iso', indent=2))

    def figure(self, name, draw):
        fig = Figure(figsize=(11, 5), layout='constrained')
        FigureCanvasAgg(fig)
        draw(fig)
        fig.savefig(self.path / f'{name}.png', dpi=140)
        fig.clear()

    def network(self, pairs, candidates):
        self.table('network_candidates', pd.DataFrame(candidates))
        if pairs is None:
            return
        pairs = pairs.copy()
        self.table('sbas_pairs', pairs)
        dates = pd.to_datetime(pd.concat([pairs.ref, pairs.rep])).value_counts().sort_index()
        self.table('acquisition_degree', pd.DataFrame({'date': dates.index, 'pair_count': dates.values}))
        self.summary['network'] = {'dates': len(dates), 'pairs': len(pairs),
                                   'single_edge_dates': int((dates == 1).sum())}
        self.save()
        def draw(fig):
            ax = fig.subplots()
            for row in pairs.itertuples():
                ax.plot(pd.to_datetime([row.ref, row.rep]),
                        [row.ref_baseline, row.rep_baseline], color='steelblue', alpha=.3, linewidth=.6)
            points = pd.concat([
                pairs[['ref', 'ref_baseline']].rename(columns={'ref': 'date', 'ref_baseline': 'baseline'}),
                pairs[['rep', 'rep_baseline']].rename(columns={'rep': 'date', 'rep_baseline': 'baseline'})]).drop_duplicates('date')
            ax.scatter(pd.to_datetime(points.date), points.baseline, s=12, color='black')
            ax.set(xlabel='Acquisition date (UTC)', ylabel='Perpendicular baseline (m)',
                   title=f'SBAS network: {len(dates)} acquisitions, {len(pairs)} pairs')
            ax.grid(alpha=.2)
        self.figure('sbas_network', draw)

    def pairs(self, phase, coherence, name):
        phase, coherence = xr.align(phase, coherence, join='exact')
        valid = np.isfinite(phase)
        stats = xr.Dataset({
            'valid_phase_pixels': valid.sum(('y', 'x')),
            'valid_phase_fraction': valid.mean(('y', 'x')),
            'mean_coherence': coherence.mean(('y', 'x')),
            'coherent_pixels': np.isfinite(coherence).sum(('y', 'x')),
        }).compute().to_dataframe().reset_index()
        self.table(name, stats)
        def draw(fig):
            axes = fig.subplots(2, 1, sharex=True)
            axes[0].plot(stats.valid_phase_fraction.to_numpy(), linewidth=.8)
            axes[0].set(ylabel='Valid phase fraction', ylim=(0, 1))
            axes[1].plot(stats.mean_coherence.to_numpy(), linewidth=.8)
            axes[1].set(xlabel='Pair index (see accompanying CSV)', ylabel='Mean accepted coherence', ylim=(0, 1))
            axes[0].set_title(name.replace('_', ' '))
        self.figure(name, draw)
        # A bounded sample: first pair and the pair with least valid phase.
        for idx in sorted({0, int(np.argmin(stats.valid_phase_fraction.to_numpy()))}):
            sample = phase.isel(pair=idx).compute()
            sample_corr = coherence.isel(pair=idx).compute()
            def draw_map(fig):
                axes = fig.subplots(1, 2)
                for ax, grid, label, options in [
                    (axes[0], sample, 'Phase (rad)', {'cmap': 'RdBu_r'}),
                    (axes[1], sample_corr, 'Accepted coherence', {'cmap': 'viridis', 'vmin': 0, 'vmax': 1}),
                ]:
                    mesh = ax.pcolormesh(grid.x, grid.y, grid.values, shading='auto', **options)
                    fig.colorbar(mesh, ax=ax, label=label)
                    ax.set(xlabel='Radar range coordinate', ylabel='Radar azimuth coordinate')
                fig.suptitle(f'{name}: {str(sample.pair.values)}')
            self.figure(f'{name}_sample_{idx:04d}', draw_map)

    def reference(self, report):
        self.table('reference_phase', report.reset_index())
        def draw(fig):
            axes = fig.subplots(2, 1, sharex=True)
            axes[0].plot(report.reference_valid_phase_pixels.to_numpy())
            axes[0].set(ylabel='Valid reference pixels', title='Combined reference support and phase offset')
            axes[1].plot(report.reference_mean_phase_rad.to_numpy())
            axes[1].set(xlabel='Pair index (see accompanying CSV)', ylabel='Mean phase offset (rad)')
        self.figure('reference_phase', draw)

    def displacement(self, displacement):
        dims = ('lat', 'lon')
        stats = xr.Dataset({
            'valid_fraction': np.isfinite(displacement).mean(dims),
            'mean_mm': displacement.mean(dims),
            'min_mm': displacement.min(dims),
            'max_mm': displacement.max(dims),
        }).compute().to_dataframe().reset_index()
        self.table('displacement_by_date', stats)
        def draw(fig):
            ax = fig.subplots()
            ax.plot(pd.to_datetime(stats.date), stats.mean_mm)
            ax.set(xlabel='Acquisition date (UTC)', ylabel='Mean cumulative LOS (mm)',
                   title='Whole output grid mean — changing valid support; not a dam-wall time series')
            ax.grid(alpha=.2)
        self.figure('displacement_by_date', draw)
