"""B1 golden regression: engine B in ALE reproduces the MAP prototype exactly.

Golden files: ``tests/golden/synthetic_crop`` (prototype f11c5f2 + stash@{0},
run unmodified on the synthetic site; see its README). Tolerance: **exact**.
The moved code performs the same float64 operations in the same order, so
arrays, series and calibrations must be bit-identical. Raster products are
compared as the prototype stored them (float32 GeoTIFF, NaN no-data).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ale_3levels.config import load_config
from ale_3levels.engines.three_level.pipeline import (
    ThreeLevelPipeline,
    inputs_from_npz,
    prototype_config,
)

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'synthetic_crop'


def _three_level(**overrides):
    config = load_config()['engine_configs']['three_level']
    return {**config, **overrides}


@pytest.fixture(scope='module')
def golden_inputs():
    return inputs_from_npz(GOLDEN / 'inputs.npz')


@pytest.fixture(scope='module')
def run(golden_inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp('three_level')
    return ThreeLevelPipeline(_three_level(), out).run(golden_inputs)


def _as_written(values: np.ndarray) -> np.ndarray:
    """The prototype's GeoTIFF writer: NaN no-data, float32."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values), values, np.nan).astype('float32')


def _json(name: str):
    return json.loads((GOLDEN / name).read_text(encoding='utf-8'))


def test_configuration_equals_the_prototype_configuration():
    golden = yaml.safe_load((GOLDEN / 'config.yaml').read_text(encoding='utf-8'))
    moved = prototype_config(_three_level())

    def without_files(section):
        return {k: v for k, v in section.items() if k != 'calibration_file'}

    for name in ('anomaly_detection', 'level2', 'level3', 'quality'):
        assert moved[name] == without_files(golden[name]), name
    dashboard = golden['monitoring']['dashboard']
    for name in (
        'spatial_coherence',
        'acceleration_cusum',
        'regime',
        'anomaly_magnitude_threshold',
    ):
        assert moved['monitoring']['dashboard'][name] == dashboard[name], name


def test_pixel_products_are_bit_identical(run):
    with np.load(GOLDEN / 'rasters.npz') as golden:
        arrays = run.arrays
        expected = {
            'anomaly_score': arrays['anomaly_score'],
            'anomaly_binary': np.where(
                np.isfinite(arrays['anomaly_score']),
                arrays['anomaly_binary'].astype(float),
                np.nan,
            ),
            'residual_z': arrays['residual_z'],
            'anomaly_l1_binary': np.where(
                np.isfinite(arrays['residual_z']),
                arrays['anomaly_l1_binary'].astype(float),
                np.nan,
            ),
            'anomaly_coherent': arrays['anomaly_coherent'].astype(float),
            'level2_maximum_instability_cusum': arrays[
                'level2_maximum_instability_cusum'
            ],
            'level2_first_alarm_index': arrays['level2_first_alarm_index'],
            'maximum_coherence_acceleration_cusum': arrays[
                'maximum_coherence_acceleration_cusum'
            ],
        }
        assert set(expected) == set(golden.files)
        for name, values in expected.items():
            np.testing.assert_array_equal(
                _as_written(values), golden[name], err_msg=name
            )


def test_series_are_identical(run):
    golden = _json('monitoring_series.json')
    moved = json.loads(run.series_paths[0].read_text(encoding='utf-8'))
    assert moved['series'] == golden['series']
    assert moved['units'] == golden['units']
    assert moved['contract_version'] == golden['contract_version']
    csv_golden = golden['series'][0].keys()
    header = run.series_paths[1].read_text(encoding='utf-8').splitlines()[0]
    assert header.split(',') == list(csv_golden)


@pytest.mark.parametrize('level', ['l1', 'l2', 'l3'])
def test_calibrations_are_identical(run, level):
    moved = json.loads(
        Path(run.metadata[f'{level}_calibration_path']).read_text(encoding='utf-8')
    )
    assert moved == _json(f'{level}_calibration.json')


def test_level_summaries_are_identical(run):
    golden = _json('monitoring_metadata.json')
    for key in (
        'level1',
        'level2',
        'level3',
        'quality',
        'lead_time',
        'run_mode',
        'run_mode_reason',
        'target_feature',
        'acquisition_dates',
        'spatial_coherence_summary',
    ):
        assert run.metadata[key] == golden[key], key
    assert run.metadata['anomaly_summary'] == _strip_units(
        _json('anomaly_summary.json')
    )
    assert run.metadata['spatial_coherence_summary'] == _json(
        'spatial_coherence_summary.json'
    )


def _strip_units(summary: dict) -> dict:
    """The prototype's anomaly_summary.json adds unit labels when writing."""
    return {
        key: value
        for key, value in summary.items()
        if key
        not in {'residual_rate_unit', 'anomaly_score_unit', 'anomaly_binary_unit'}
    }


def test_legacy_global_zscore_is_identical(golden_inputs, tmp_path):
    result = ThreeLevelPipeline(
        _three_level(zscore_formula='legacy_global'), tmp_path
    ).run(golden_inputs)
    with np.load(GOLDEN / 'legacy_za.npz') as golden:
        np.testing.assert_array_equal(
            result.arrays['legacy_global_score'], golden['score']
        )
        np.testing.assert_array_equal(
            result.arrays['legacy_global_binary'], golden['binary']
        )
    assert result.metadata['legacy_global_summary'] == _json('legacy_za_summary.json')


def test_run_is_deterministic(golden_inputs, run, tmp_path):
    again = ThreeLevelPipeline(_three_level(), tmp_path).run(golden_inputs)
    assert set(again.arrays) == set(run.arrays)
    for name, values in run.arrays.items():
        np.testing.assert_array_equal(again.arrays[name], values, err_msg=name)
    assert again.series_paths[0].read_bytes() == run.series_paths[0].read_bytes()


def test_unknown_zscore_formula_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='zscore_formula'):
        ThreeLevelPipeline(_three_level(zscore_formula='other'), tmp_path)
