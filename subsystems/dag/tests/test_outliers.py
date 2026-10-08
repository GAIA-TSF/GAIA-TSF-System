import numpy as np
import pytest

from subsystems.dag.utils.outliers import transform_outliers


def test_outlier_transformation_is_disabled_by_default():
    features = {'a': np.array([0.0, 10.0, 1000.0], dtype=np.float32)}
    assert transform_outliers(features, {'method': 'log'}) is features


def test_signed_log1p_compresses_positive_and_negative_outliers():
    values = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0, np.nan])
    result = transform_outliers(
        {'deformation': values},
        {'enabled': True, 'method': 'log', 'signed_log': True},
    )['deformation']
    expected = np.sign(values[:5]) * np.log1p(np.abs(values[:5]))
    assert np.allclose(result[:5], expected)
    assert np.isnan(result[5])
    assert abs(result[4]) < abs(values[4])


def test_log_transform_can_target_selected_features_only():
    features = {
        'precipitation': np.array([0.0, 9.0]),
        'temperature': np.array([0.0, 9.0]),
    }
    result = transform_outliers(
        features,
        {'enabled': True, 'method': 'log', 'features': ['precipitation']},
    )
    assert np.allclose(result['precipitation'], np.log1p([0.0, 9.0]))
    assert result['temperature'] is features['temperature']


def test_unsigned_log_rejects_negative_values():
    with pytest.raises(ValueError, match='negative values'):
        transform_outliers(
            {'a': np.array([-1.0, 2.0])},
            {'enabled': True, 'method': 'log', 'signed_log': False},
        )


def test_quantile_clip_is_supported_by_same_preprocessing_stage():
    result = transform_outliers(
        {'a': np.array([0.0, 1.0, 2.0, 100.0])},
        {'enabled': True, 'method': 'clip', 'clip_range': [0.0, 0.75]},
    )['a']
    assert result[-1] < 100.0
    assert result[0] == 0.0
