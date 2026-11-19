"""Golden prototype set: integrity (fast) and contract equivalence (slow, local).

The slow test checks that the contract reproduces the golden prototype inputs
bit for bit. Set ``ALE_SYNTHETIC_TABLE`` to the synthetic site's MAP
``results/forecasts/slope_trf/residuals.npz`` and ``ALE_SYNTHETIC_MASK`` to its
``static/tsf_mask.tif``. Skipped when the data is not available.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from ale_3levels.contract import load_contract, read_mask

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'synthetic_crop'
TABLE = os.environ.get('ALE_SYNTHETIC_TABLE')
MASK = os.environ.get('ALE_SYNTHETIC_MASK')

HAVE_DATA = bool(TABLE and MASK and Path(TABLE).is_file() and Path(MASK).is_file())


def test_golden_set_is_intact():
    manifest = json.loads((GOLDEN / 'manifest.json').read_text())
    for name in manifest['files']:
        assert (GOLDEN / name).is_file(), name
    digest = hashlib.sha256((GOLDEN / 'inputs.npz').read_bytes()).hexdigest()
    assert digest == manifest['input_sha256']
    assert manifest['source_unit'] == 'm' and manifest['lead'] == 1


@pytest.mark.slow
@pytest.mark.skipif(
    not HAVE_DATA,
    reason='synthetic site data not available (set ALE_SYNTHETIC_TABLE/MASK)',
)
def test_contract_matches_golden_prototype_inputs():
    manifest = json.loads((GOLDEN / 'manifest.json').read_text())
    r0, r1, c0, c1 = manifest['crop_rows_cols']
    contract = load_contract(
        Path(TABLE), lead=1, source_units_override='m', mask=read_mask(Path(MASK))
    )
    window = (slice(None), slice(r0, r1), slice(c0, c1))
    with np.load(GOLDEN / 'inputs.npz') as golden:
        assert contract.dates.tolist() == golden['dates'].tolist()
        assert (
            contract.forecast_origin.tolist()
            == golden['forecast_origin_dates'].tolist()
        )
        for field, key in (
            ('observed_mm_per_day', 'observed_stack'),
            ('predicted_mm_per_day', 'prediction_stack'),
            ('residual_mm_per_day', 'residual_stack'),
        ):
            np.testing.assert_array_equal(getattr(contract, field)[window], golden[key])
        np.testing.assert_array_equal(contract.mask[r0:r1, c0:c1], golden['mask'])
        start = int(golden['monitoring_window'][0])
        assert contract.monitoring_start_index == start
