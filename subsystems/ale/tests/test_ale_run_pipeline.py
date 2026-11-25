"""Project runner scripts/run_pipeline.py: separate processes, no imports."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from ale_3levels.demo import write_demo_site

RUNNER = Path(__file__).resolve().parents[3] / 'scripts' / 'run_pipeline.py'


def test_runner_imports_neither_map_nor_ale():
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not {m for m in modules if m.startswith(('subsystems', 'ale_3levels'))}


def test_runner_runs_ale_and_report_on_existing_map_output(tmp_path):
    demo = write_demo_site(tmp_path / 'site')
    out = tmp_path / 'out'
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            '--skip-map',
            '--ale-config',
            str(demo['config']),
            '--out',
            str(out),
            '--truth',
            str(demo['truth_mask']),
        ],
        check=True,
        capture_output=True,
    )
    assert (out / 'fixed_gates' / 'alerts.npz').is_file()
    assert (out / 'three_level' / 'alerts.npz').is_file()
    assert (out / 'report' / 'report.md').is_file()


def test_runner_requires_its_inputs():
    result = subprocess.run(
        [sys.executable, str(RUNNER), '--skip-map'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert '--ale-config and --out are required' in result.stderr
