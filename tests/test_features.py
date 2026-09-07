import numpy as np

from sleep_stage_classification.features import extract_epoch_features, hjorth_parameters, permutation_entropy
from sleep_stage_classification.preprocessing import epoch_signal, zscore_per_channel


def test_epoch_signal_shape():
    signal = np.arange(2 * 3000, dtype=float).reshape(2, 3000)
    epochs = epoch_signal(signal, sample_rate_hz=100, epoch_seconds=30)
    assert epochs.shape == (1, 2, 3000)


def test_zscore_per_channel_centers_each_channel():
    signal = np.array([[1, 2, 3], [10, 20, 30]], dtype=float)
    scaled = zscore_per_channel(signal)
    assert np.allclose(scaled.mean(axis=1), 0)
    assert np.allclose(scaled.std(axis=1), 1)


def test_feature_extraction_contains_expected_keys():
    epoch = np.vstack([np.sin(np.linspace(0, 8 * np.pi, 3000)), np.cos(np.linspace(0, 4 * np.pi, 3000))])
    features = extract_epoch_features(epoch, ["EEG Fpz-Cz", "EOG horizontal"], sample_rate_hz=100)
    assert "EEG_Fpz-Cz_delta_power" in features
    assert "EOG_horizontal_tkeo" in features
    assert 0 <= features["EEG_Fpz-Cz_permutation_entropy"] <= 1


def test_hjorth_and_entropy_are_finite():
    x = np.sin(np.linspace(0, 2 * np.pi, 300))
    mobility, complexity = hjorth_parameters(x)
    assert mobility > 0
    assert complexity > 0
    assert 0 <= permutation_entropy(x) <= 1
