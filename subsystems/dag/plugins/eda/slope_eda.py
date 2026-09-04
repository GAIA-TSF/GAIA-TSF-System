"""Exploratory statistics, plots, and maps for Sentinel-1 LOS time series."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from subsystems.dag.core.interfaces import Plugin
from subsystems.dag.utils import plotting
from subsystems.dag.utils.raster import temporal_mean, temporal_std
from subsystems.dag.utils.statistics import finite_values, time_series_statistics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EDAOutputPaths:
    """Output paths produced by slope EDA."""

    statistics: Path
    temporal_mean_std: Path
    histogram: Path
    boxplot: Path
    mean_heatmap: Path
    std_heatmap: Path


@dataclass(frozen=True)
class SlopeEDAResult:
    """Slope EDA outputs."""

    statistics: dict[str, object]
    paths: EDAOutputPaths
    mean_map: np.ndarray
    std_map: np.ndarray


class SlopeEDA(Plugin):
    """Compute descriptive statistics and plots for LOS deformation."""

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return 'slope_eda'

    def run(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
        output_dir: Path,
        options: dict[str, object],
    ) -> SlopeEDAResult:
        """Run EDA over a masked LOS raster time series.

        Args:
            data: Masked LOS stack with shape ``(time, rows, cols)``.
            dates: Acquisition dates.
            output_dir: Directory where EDA outputs are written.
            options: EDA options from configuration.

        Returns:
            Statistics, output paths, and computed mean/std maps.
        """
        if data.size == 0:
            raise ValueError('Empty LOS dataset.')
        if len(dates) == 0:
            raise ValueError('No acquisition dates were provided.')

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_options = self._plot_options(options)
        histogram_bins = int(options['histogram_bins'])
        output_paths = self._output_paths(output_dir, options)

        statistics = time_series_statistics(data, dates, histogram_bins)
        mean_map = temporal_mean(data)
        std_map = temporal_std(data)

        self._write_statistics(statistics, output_paths.statistics)
        self._write_figures(
            data=data,
            dates=dates,
            output_paths=output_paths,
            mean_map=mean_map,
            std_map=std_map,
            histogram_bins=histogram_bins,
            plot_options=plot_options,
        )

        LOGGER.info('Slope EDA outputs written to %s.', output_dir)
        return SlopeEDAResult(
            statistics=statistics,
            paths=output_paths,
            mean_map=mean_map,
            std_map=std_map,
        )

    def _plot_options(self, options: dict[str, object]) -> dict[str, object]:
        return {
            'dpi': int(options.get('dpi', 300)),
            'style': options.get('style'),
            'cmap': str(options.get('cmap', 'viridis')),
        }

    def _output_paths(
        self,
        output_dir: Path,
        options: dict[str, object],
    ) -> EDAOutputPaths:
        filenames = options['filenames']
        if not isinstance(filenames, dict):
            raise TypeError('EDA filenames configuration must be a mapping.')
        return EDAOutputPaths(
            statistics=output_dir / str(filenames['statistics']),
            temporal_mean_std=output_dir / str(filenames['temporal_mean_std']),
            histogram=output_dir / str(filenames['histogram']),
            boxplot=output_dir / str(filenames['boxplot']),
            mean_heatmap=output_dir / str(filenames['mean_heatmap']),
            std_heatmap=output_dir / str(filenames['std_heatmap']),
        )

    def _write_statistics(
        self,
        statistics: dict[str, object],
        output_path: Path,
    ) -> None:
        with output_path.open('w', encoding='utf-8') as file:
            json.dump(statistics, file, indent=2)

    def _write_figures(
        self,
        data: np.ndarray,
        dates: tuple[date, ...],
        output_paths: EDAOutputPaths,
        mean_map: np.ndarray,
        std_map: np.ndarray,
        histogram_bins: int,
        plot_options: dict[str, object],
    ) -> None:
        finite = finite_values(data)
        per_acquisition_means = np.nanmean(data, axis=(1, 2))
        per_acquisition_stds = np.nanstd(data, axis=(1, 2))
        dpi = int(plot_options['dpi'])
        style = plot_options['style']
        cmap = str(plot_options['cmap'])

        plotting.save_temporal_mean_std_plot(
            dates,
            per_acquisition_means,
            per_acquisition_stds,
            output_paths.temporal_mean_std,
            dpi=dpi,
            style=style if isinstance(style, str) else None,
        )
        plotting.save_histogram(
            finite,
            histogram_bins,
            output_paths.histogram,
            dpi=dpi,
            style=style if isinstance(style, str) else None,
        )
        plotting.save_boxplot(
            data,
            dates,
            output_paths.boxplot,
            dpi=dpi,
            style=style if isinstance(style, str) else None,
        )
        plotting.save_heatmap(
            mean_map,
            output_paths.mean_heatmap,
            title='Mean LOS',
            colorbar_label='Mean LOS displacement',
            dpi=dpi,
            cmap=cmap,
            style=style if isinstance(style, str) else None,
        )
        plotting.save_heatmap(
            std_map,
            output_paths.std_heatmap,
            title='Temporal standard deviation',
            colorbar_label='LOS standard deviation',
            dpi=dpi,
            cmap=cmap,
            style=style if isinstance(style, str) else None,
        )
