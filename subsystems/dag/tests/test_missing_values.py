import numpy as np
import pytest

from subsystems.dag.utils.missing_values import handle_missing_values


def test_missing_value_handling_is_disabled_by_default():
    features = {'a': np.array([1.0, np.nan], dtype=np.float32)}
    assert handle_missing_values(features, {'strategy': 'median'}) is features


@pytest.mark.parametrize(
    ('strategy', 'expected'),
    [('mean', 3.0), ('median', 3.0)],
)
def test_imputes_non_finite_values(strategy, expected):
    result = handle_missing_values(
        {'a': np.array([1.0, np.nan, 5.0, np.inf])},
        {'enabled': True, 'strategy': strategy, 'max_nan_ratio': 0.5},
    )
    assert np.allclose(result['a'], [1.0, expected, 5.0, expected])


def test_drop_marks_incomplete_positions_missing_across_features():
    result = handle_missing_values(
        {
            'a': np.array([1.0, np.nan, 3.0]),
            'b': np.array([4.0, 5.0, 6.0]),
        },
        {'enabled': True, 'strategy': 'drop', 'max_nan_ratio': 1.0},
    )
    assert np.isnan(result['a'][1])
    assert np.isnan(result['b'][1])
    assert np.allclose(result['b'][[0, 2]], [4.0, 6.0])


def test_imputation_does_not_fill_outside_valid_raster_mask():
    result = handle_missing_values(
        {'a': np.array([[1.0, np.nan], [3.0, np.nan]])},
        {'enabled': True, 'strategy': 'mean', 'max_nan_ratio': 0.5},
        valid_mask=np.array([[1, 1], [1, 0]], dtype=bool),
    )['a']
    assert np.isclose(result[0, 1], 2.0)
    assert np.isnan(result[1, 1])


def test_rejects_feature_above_configured_missing_ratio():
    with pytest.raises(ValueError, match='exceeds configured maximum'):
        handle_missing_values(
            {'a': np.array([1.0, np.nan, np.nan])},
            {'enabled': True, 'strategy': 'median', 'max_nan_ratio': 0.5},
        )
