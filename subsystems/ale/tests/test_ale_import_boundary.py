"""MAP and ALE must not import each other; ALE reads saved MAP outputs only."""

from __future__ import annotations

import ast
from pathlib import Path

ALE_ROOT = Path(__file__).resolve().parents[1]
MAP_ROOT = ALE_ROOT.parent / 'map'
ALE_MODULES = ('ale_3levels', 'subsystems.ale')


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _offenders(root: Path, forbidden) -> dict[str, list[str]]:
    found = {}
    for path in sorted(root.rglob('*.py')):
        names = sorted(name for name in _imports(path) if forbidden(name))
        if names:
            found[str(path.relative_to(root))] = names
    return found


def _is_map_or_other_subsystem(name: str) -> bool:
    return name == 'subsystems' or name.startswith('subsystems.')


def _is_ale(name: str) -> bool:
    return any(
        name == module or name.startswith(f'{module}.') for module in ALE_MODULES
    )


def test_layout():
    assert (ALE_ROOT / 'ale_3levels' / 'contract.py').is_file()
    assert (MAP_ROOT / 'pipelines' / 'inference_pipeline.py').is_file()


def test_ale_does_not_import_map_or_other_subsystems():
    assert _offenders(ALE_ROOT, _is_map_or_other_subsystem) == {}


def test_map_does_not_import_ale():
    assert _offenders(MAP_ROOT, _is_ale) == {}


def test_boundary_check_detects_violations(tmp_path):
    (tmp_path / 'a.py').write_text('from subsystems.map.utils import forecasts\n')
    (tmp_path / 'b.py').write_text(
        'import ale_3levels.contract\nimport subsystems.ale\n'
    )
    assert _offenders(tmp_path, _is_map_or_other_subsystem) == {
        'a.py': ['subsystems.map.utils'],
        'b.py': ['subsystems.ale'],
    }
    assert _offenders(tmp_path, _is_ale) == {
        'b.py': ['ale_3levels.contract', 'subsystems.ale'],
    }
