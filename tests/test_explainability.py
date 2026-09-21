import numpy as np

from sleep_stage_classification.explainability import _as_sample_feature_class_values


def test_shap_value_normalization_keeps_sample_feature_class_order():
    values = np.arange(24, dtype=float).reshape(2, 3, 4)
    normalized = _as_sample_feature_class_values(values, sample_count=2, feature_count=3)
    assert normalized.shape == (2, 3, 4)
    assert np.array_equal(normalized, values)


def test_shap_value_normalization_transposes_class_feature_order():
    values = np.arange(24, dtype=float).reshape(2, 4, 3)
    normalized = _as_sample_feature_class_values(values, sample_count=2, feature_count=3)
    assert normalized.shape == (2, 3, 4)
    assert np.array_equal(normalized, np.transpose(values, (0, 2, 1)))
