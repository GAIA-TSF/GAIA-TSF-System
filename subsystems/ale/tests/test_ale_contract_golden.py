"""Golden prototype set: integrity (fast) and contract equivalence (slow, local).

The slow test checks that the contract reproduces the golden prototype inputs
from a fresh MAP inference run: observations bit for bit, predictions to
``rtol=1e-10``. MAP's random forest predicts with ``n_jobs > 1`` and sums the
trees in thread order, so repeated MAP runs differ by ~1e-14 mm/day (measured
2026-11-25: 401 of 118784 values, max 9.3e-15). The contract itself is
deterministic (same table -> identical stacks). Set ``ALE_SYNTHETIC_TABLE`` to the synthetic site's MAP
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
        np.testing.assert_array_equal(
            contract.observed_mm_per_day[window], golden['observed_stack']
        )
        for field, key in (
            ('predicted_mm_per_day', 'prediction_stack'),
            ('residual_mm_per_day', 'residual_stack'),
        ):
            np.testing.assert_allclose(
                getattr(contract, field)[window], golden[key], rtol=1e-10, atol=1e-12
            )
        np.testing.assert_array_equal(contract.mask[r0:r1, c0:c1], golden['mask'])
        start = int(golden['monitoring_window'][0])
        assert contract.monitoring_start_index == start
