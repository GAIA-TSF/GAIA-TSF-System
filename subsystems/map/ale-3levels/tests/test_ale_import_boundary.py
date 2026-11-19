"""MAP and ALE must not import each other (ALE moves to subsystems/ale later)."""

from __future__ import annotations

import ast
from pathlib import Path

ALE_ROOT = Path(__file__).resolve().parents[1]
MAP_ROOT = ALE_ROOT.parent
ALE_PACKAGE = ALE_ROOT / 'ale_3levels'


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _python_files(root: Path, exclude: Path | None = None) -> list[Path]:
    return [
        path
        for path in root.rglob('*.py')
        if exclude is None or exclude not in path.parents
    ]


def test_ale_does_not_import_map_or_other_subsystems():
    offenders = {
        str(path.relative_to(ALE_ROOT)): sorted(
            name for name in _imports(path) if name.split('.')[0] == 'subsystems'
        )
        for path in _python_files(ALE_ROOT)
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_map_does_not_import_ale():
    offenders = {
        str(path.relative_to(MAP_ROOT)): sorted(
            name for name in _imports(path) if name.split('.')[0] == 'ale_3levels'
        )
        for path in _python_files(MAP_ROOT, exclude=ALE_ROOT)
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_boundary_check_detects_violations(tmp_path):
    sample = tmp_path / 'sample.py'
    sample.write_text(
        'from subsystems.map.utils import forecasts\nimport ale_3levels.contract\n'
    )
    assert _imports(sample) == {'subsystems.map.utils', 'ale_3levels.contract'}
