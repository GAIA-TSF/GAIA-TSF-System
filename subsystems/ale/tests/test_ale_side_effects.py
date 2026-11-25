"""No side effects (spec rule 4) for whole ALE runs, engine B included.

Each command may write only into its own output directory: alerts, sectors,
engine-B calibration and state. No network, process, shell command, or write
anywhere else (MAP inputs, frozen calibration or previous state it reads, the
working directory); temporary files must be removed. Engine A's scoring alone
is also checked in test_ale_fixed_gates.py.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import tempfile

import pytest
from ale_3levels.demo import write_demo_site
from ale_3levels.run import calibrate, run


def _snapshot(root):
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Demo site, empty working and temp directories, no network or processes."""
    site = write_demo_site(tmp_path / 'site', n_dates=50)
    workdir, scratch = tmp_path / 'cwd', tmp_path / 'tmp'
    workdir.mkdir()
    scratch.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError(f'side effect attempted: {args!r}')

    monkeypatch.chdir(workdir)
    monkeypatch.setattr(tempfile, 'tempdir', str(scratch))
    for module, name in (
        (socket, 'socket'),
        (socket, 'create_connection'),
        (socket, 'getaddrinfo'),
        (subprocess, 'Popen'),
        (os, 'system'),
        (os, 'fork'),
        (os, 'execv'),
        (os, 'execve'),
    ):
        monkeypatch.setattr(module, name, forbidden)

    def step(target, command):
        """Run ``command``; return the files it wrote, all inside ``target``."""
        before = _snapshot(tmp_path)
        command()
        after = _snapshot(tmp_path)
        assert set(before) <= set(after), 'files were deleted'
        written = {p for p in after if before.get(p) != after[p]}
        assert {p for p in written if not p.is_relative_to(target)} == set()
        assert list(workdir.iterdir()) == []
        assert list(scratch.iterdir()) == []
        return {p.relative_to(target) for p in written}

    return tmp_path, site, step


@pytest.mark.parametrize('engine', ['three_level', 'fixed_gates', 'both'])
def test_run_writes_only_into_its_engine_directories(sandbox, engine):
    root, site, step = sandbox
    out = root / 'out'
    written = step(out, lambda: run(engine, out, config_path=site['config']))
    tops = {p.parts[0] for p in written}
    expected = {'fixed_gates', 'three_level'} if engine == 'both' else {engine}
    assert tops == expected | {'run.json'}
    if engine != 'fixed_gates':
        # Engine B keeps its CUSUM state inside its own directory.
        assert any(p.parts[:2] == ('three_level', 'state') for p in written)


def test_engine_b_calibrate_run_and_continue_stay_in_their_outputs(sandbox):
    root, site, step = sandbox
    config = site['config']

    frozen = root / 'frozen'
    written = step(frozen, lambda: calibrate(frozen, config_path=config))
    assert {p.parts[0] for p in written} == {'calibration', 'calibration_manifest.json'}

    # Reads the frozen calibration, must not modify it.
    first = root / 'first'
    step(
        first,
        lambda: run(
            'three_level',
            first,
            config_path=config,
            calibration_dir=frozen / 'calibration',
        ),
    )

    # Reads the previous state, must not modify it.
    second = root / 'second'
    step(
        second,
        lambda: run(
            'three_level',
            second,
            config_path=config,
            calibration_dir=frozen / 'calibration',
            state_dir=first / 'three_level' / 'state',
        ),
    )
