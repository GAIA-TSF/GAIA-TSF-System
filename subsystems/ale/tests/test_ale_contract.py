"""Contract reader: validation, harmonization to mm and mm/day, splits."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
from ale_3levels.contract import (
    CALIBRATION,
    MONITORING,
    TRAIN,
    VALIDATION,
    ContractError,
    load_contract,
    summary,
)
from ale_fixtures import (
    DATES,
    HEIGHT,
    MONITORING_START,
    OFFSETS,
    WIDTH,
    cumulative_m,
    site_table,
    write_map_table,
)


def test_harmonizes_cumulative_and_rate_fields_for_lead_one(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table, source_units_override='m')
    assert contract.dates.tolist() == list(DATES[1:])
    assert contract.forecast_origin.tolist() == list(DATES[:-1])
    t, row, col = 3, 2, 1  # target DATES[4] (after the 24-day gap), origin DATES[3]
    dt = OFFSETS[4] - OFFSETS[3]
    assert dt == 24
    assert contract.delta_t_days[t, row, col] == dt
    observed = cumulative_m(4, row, col)
    persistence = cumulative_m(3, row, col)
    predicted = persistence - 0.0009
    assert contract.observed_mm[t, row, col] == observed * 1000.0
    assert contract.persistence_mm[t, row, col] == persistence * 1000.0
    assert contract.predicted_mm[t, row, col] == predicted * 1000.0
    assert contract.residual_mm[t, row, col] == (observed - predicted) * 1000.0
    assert (
        contract.observed_mm_per_day[t, row, col]
        == (observed - persistence) / dt * 1000.0
    )
    assert (
        contract.predicted_mm_per_day[t, row, col]
        == (predicted - persistence) / dt * 1000.0
    )
    assert (
        contract.residual_mm_per_day[t, row, col]
        == (observed - predicted) / dt * 1000.0
    )
    assert contract.attrs['source_units'] == 'm'
    assert contract.attrs['declared_units'] == 'mm'
    assert contract.attrs['units'] == {
        'cumulative': 'mm',
        'rate': 'mm/day',
        'delta_t': 'day',
    }


def test_declared_mm_is_not_rescaled(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table)
    assert contract.attrs['source_units'] == 'mm'
    assert contract.observed_mm[0, 0, 0] == cumulative_m(1, 0, 0)


def test_lead_two_uses_its_own_interval(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table, lead=2, source_units_override='m')
    assert contract.dates.tolist() == list(DATES[2:])
    # target DATES[4] from origin DATES[2]: 12 + 24 days.
    t = contract.dates.tolist().index(DATES[4])
    assert contract.delta_t_days[t, 0, 0] == 36
    assert contract.forecast_origin[t] == DATES[2]


def test_split_combines_period_and_sample_role(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table)
    by_date = {d: int(contract.split[i, 0, 0]) for i, d in enumerate(contract.dates)}
    assert by_date[DATES[1]] == TRAIN
    assert by_date[DATES[3]] == VALIDATION
    assert by_date[DATES[4]] == CALIBRATION  # sample_role test
    assert by_date[DATES[5]] == CALIBRATION
    assert all(by_date[d] == MONITORING for d in DATES[6:])
    assert contract.monitoring_start_index == contract.dates.tolist().index(
        MONITORING_START
    )
    assert summary(contract)['calibration_split_dates'] == 2


def test_expected_rows_are_excluded(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table, lead=2)
    assert contract.dates[-1] == DATES[-1]
    assert np.all(np.isfinite(contract.observed_mm))


def test_observed_series_includes_first_origin(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table, source_units_override='m')
    assert contract.series_dates.tolist() == list(DATES)
    expected = np.array(
        [
            [
                [cumulative_m(i, r, c) * 1000.0 for c in range(WIDTH)]
                for r in range(HEIGHT)
            ]
            for i in range(len(DATES))
        ]
    )
    np.testing.assert_array_equal(contract.observed_series_mm, expected)
    assert contract.attrs['reference_date'] == DATES[0]


def test_mask_and_validity(tmp_path):
    map_table = site_table(tmp_path)
    mask = np.ones((HEIGHT, WIDTH), dtype=bool)
    mask[0, 0] = False
    contract = load_contract(map_table, mask=mask)
    assert not contract.valid_mask[:, 0, 0].any()
    assert contract.valid_mask[:, 1, 1].all()
    with pytest.raises(ContractError, match='Mask shape'):
        load_contract(map_table, mask=np.ones((2, 2), dtype=bool))


def test_identity_attributes(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table)
    assert contract.attrs['site_id'] == 'site_a'
    assert contract.attrs['model_id'] == 'trf'
    assert contract.attrs['model_version'] == 'slope_trf@test'
    assert contract.attrs['tsf_version_provisional'] is True
    assert contract.attrs['tsf_version'].startswith('unversioned-')
    assert contract.attrs['los_sign_convention'] == 'negative_away_from_satellite'


def test_explicit_tsf_version_is_used(tmp_path):
    path = write_map_table(tmp_path, sidecar_updates={'tsf_version': 'dpr-2026-11-01'})
    contract = load_contract(path)
    assert contract.attrs['tsf_version'] == 'dpr-2026-11-01'
    assert contract.attrs['tsf_version_provisional'] is False


def test_model_mismatch_is_rejected(tmp_path):
    map_table = site_table(tmp_path)
    with pytest.raises(ContractError, match="selects 'lstm'"):
        load_contract(map_table, model_id='lstm')


@pytest.mark.parametrize(
    ('updates', 'message'),
    [
        ({'schema_version': '1.0'}, 'Unsupported MAP table schema_version'),
        ({'target_units': 'furlong'}, 'Cannot convert MAP unit'),
        ({'target_variable': 'velocity'}, 'not cumulative LOS'),
    ],
)
def test_sidecar_errors(tmp_path, updates, message):
    path = write_map_table(tmp_path, sidecar_updates=updates)
    with pytest.raises(ContractError, match=message):
        load_contract(path)


def test_missing_sidecar_key(tmp_path):
    path = write_map_table(tmp_path)
    sidecar = json.loads(path.with_suffix('.json').read_text())
    del sidecar['model_version']
    path.with_suffix('.json').write_text(json.dumps(sidecar))
    with pytest.raises(ContractError, match='model_version'):
        load_contract(path)


def test_missing_files(tmp_path):
    with pytest.raises(ContractError, match='not found'):
        load_contract(tmp_path / 'residuals.npz')


def _drop_residual(arrays):
    del arrays['residual']


def _break_series_id(arrays):
    arrays['series_id'] = arrays['series_id'] + 1


def _duplicate_rows(arrays):
    for name in arrays:
        arrays[name] = np.concatenate([arrays[name], arrays[name][:1]])


def _bad_role(arrays):
    arrays['sample_role'] = arrays['sample_role'].astype('<U12')
    arrays['sample_role'][0] = 'holdout'


def _bad_date(arrays):
    arrays['last_observation_date'] = arrays['last_observation_date'].astype('<U12')
    arrays['last_observation_date'][0] = 'yesterday'


def _non_positive_interval(arrays):
    arrays['last_observation_date'] = arrays['target_date'].astype('<U10')


@pytest.mark.parametrize(
    ('update', 'message'),
    [
        (_drop_residual, 'lacks required columns'),
        (_break_series_id, 'series_id'),
        (_duplicate_rows, 'duplicate'),
        (_bad_role, 'sample_role'),
        (_bad_date, 'not an ISO date'),
        (_non_positive_interval, 'after last_observation_date'),
    ],
)
def test_column_errors(tmp_path, update, message):
    path = write_map_table(tmp_path, columns_update=update)
    with pytest.raises(ContractError, match=message):
        load_contract(path)


def test_inconsistent_observations_are_rejected(tmp_path):

    def shift_persistence(arrays):
        arrays['persistence_value'] = arrays['persistence_value'] + 0.5

    path = write_map_table(tmp_path, columns_update=shift_persistence)
    with pytest.raises(ContractError, match='disagree'):
        load_contract(path)


def test_reading_does_not_modify_map_output(tmp_path):
    map_table = site_table(tmp_path)
    before = {p: p.read_bytes() for p in map_table.parent.iterdir()}
    load_contract(map_table)
    assert {p: p.read_bytes() for p in map_table.parent.iterdir()} == before


def test_dates_are_iso_and_sorted(tmp_path):
    map_table = site_table(tmp_path)
    contract = load_contract(map_table)
    parsed = [date.fromisoformat(d) for d in contract.dates]
    assert parsed == sorted(parsed)
