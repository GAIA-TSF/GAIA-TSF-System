import numpy as np

from subsystems.dag.utils.normalization import normalize_features


def test_normalization_is_disabled_by_default():
    features = {'a': np.array([1.0, 2.0, np.nan], dtype=np.float32)}
    result = normalize_features(features, {'method': 'minmax'})
    assert result is features


def test_minmax_normalizes_each_feature_and_preserves_nan():
    features = {
        'small': np.array([1.0, 2.0, 3.0, np.nan]),
        'large': np.array([100.0, 200.0, 300.0, np.nan]),
    }
    result = normalize_features(
        features,
        {'enabled': True, 'method': 'minmax', 'per_feature': True},
    )
    assert np.allclose(result['small'][:3], [0.0, 0.5, 1.0])
    assert np.allclose(result['large'][:3], [0.0, 0.5, 1.0])
    assert np.isnan(result['small'][3])


def test_zscore_produces_zero_mean_and_unit_standard_deviation():
    result = normalize_features(
        {'feature': np.array([1.0, 2.0, 3.0, 4.0])},
        {'enabled': True, 'method': 'zscore'},
    )['feature']
    assert np.isclose(np.mean(result), 0.0)
    assert np.isclose(np.std(result), 1.0)


def test_constant_feature_normalizes_to_zero():
    result = normalize_features(
        {'constant': np.full((2, 2), 7.0)},
        {'enabled': True, 'method': 'minmax'},
    )['constant']
    assert np.array_equal(result, np.zeros((2, 2), dtype=np.float32))
