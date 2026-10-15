"""Standalone AMD index feature-generation stage."""

from __future__ import annotations

from pathlib import Path

from subsystems.dag.pipelines.amd_eda_pipeline import AMDEDAPipeline


class AMDIndexPipeline(AMDEDAPipeline):
    """Generate ``amd_index.tif`` and its metadata without running EDA."""

    def run(self):
        result = self._run_index_generation()
        return {
            'pipeline': 'amd_index',
            'acquisitions': result['acquisitions'],
            'index': result['index'],
            'output_dir': result['output_dir'],
        }
