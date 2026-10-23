import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from subsystems.dpr.preprocessing_pipelines.insar_checkpoints import PairCheckpoints


def sample():
    phase = xr.DataArray(np.array([[[1, np.nan]], [[-2, 3]]], dtype='float32'),
                         dims=('pair', 'y', 'x'),
                         coords={'pair': ['a b', 'b c'], 'y': [12.5], 'x': [3., 6.],
                                 'ref': ('pair', pd.to_datetime(['2020-01-01', '2020-01-13'])),
                                 'rep': ('pair', pd.to_datetime(['2020-01-13', '2020-01-25']))})
    phase.attrs['units'] = 'rad'
    return xr.Dataset({'wrapped_phase': phase, 'coherence': xr.ones_like(phase) * .8,
                       'valid_mask': phase.notnull().astype('uint8')})


def test_lossless_roundtrip_and_network(tmp_path):
    checkpoint = PairCheckpoints(tmp_path, {})
    wrapped = sample()
    checkpoint.network(pd.DataFrame({'ref': ['a', 'b'], 'rep': ['b', 'c']}), [{'days': 12}])
    checkpoint.write('wrapped', wrapped)
    unwrapped = xr.Dataset({'unwrapped_phase': wrapped.wrapped_phase * 2,
                            'snaphu_component': xr.full_like(wrapped.wrapped_phase, 2).where(wrapped.valid_mask)})
    checkpoint.write('unwrapped', unwrapped)
    for stage, expected in [('wrapped', wrapped), ('unwrapped', unwrapped)]:
        with PairCheckpoints.open_stage(checkpoint.path, stage) as reopened:
            xr.testing.assert_identical(reopened.compute(), expected)
    assert json.loads((checkpoint.path / 'sbas_pairs.json').read_text())[1]['rep'] == 'c'
    assert checkpoint.manifest['pairs'][0]['wrapped_phase_finite_pixels'] == 1


def test_failed_write_preserves_committed_pairs(tmp_path, monkeypatch):
    checkpoint = PairCheckpoints(tmp_path, {})
    original = xr.Dataset.to_netcdf
    def failing_write(data, path, **kwargs):
        if '000001' in str(path):
            path.write_bytes(b'partial')
            raise OSError('disk full')
        return original(data, path, **kwargs)
    monkeypatch.setattr(xr.Dataset, 'to_netcdf', failing_write)
    with pytest.raises(OSError, match='disk full'):
        checkpoint.write('wrapped', sample())
    manifest = json.loads((checkpoint.path / 'manifest.json').read_text())
    assert (checkpoint.path / manifest['pairs'][0]['wrapped']).is_file()
    assert 'wrapped' not in manifest['pairs'][1]
    assert not list(checkpoint.path.rglob('*.tmp'))
    with pytest.raises(ValueError, match='incomplete'):
        PairCheckpoints.open_stage(checkpoint.path, 'wrapped')


def test_reordered_pair_stack_is_rejected(tmp_path):
    checkpoint = PairCheckpoints(tmp_path, {})
    checkpoint.write('wrapped', sample())
    with pytest.raises(ValueError, match='order differs'):
        checkpoint.write('unwrapped', sample().isel(pair=[1, 0]))
