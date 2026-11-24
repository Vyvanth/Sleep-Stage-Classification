import numpy as np

import pandas as pd

from sleep_stage_classification.features import (
    add_sequential_context,
    extract_epoch_features,
    hjorth_parameters,
    permutation_entropy,
    spectral_features,
)
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
    assert "EOG_horizontal_tkeo_mean" in features
    assert 0 <= features["EEG_Fpz-Cz_permutation_entropy"] <= 1


def test_feature_extraction_has_requested_feature_families_for_standard_four_channel_recording():
    epoch = np.arange(4 * 3000, dtype=float).reshape(4, 3000)
    features = extract_epoch_features(
        epoch,
        ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"],
        sample_rate_hz=100,
    )

    assert len(features) == 116
    assert "EEG_Fpz-Cz_gamma_power" in features
    assert "EEG_Fpz-Cz_hjorth_activity" in features
    assert "EEG_Fpz-Cz_permutation_entropy" in features
    assert "EOG_horizontal_tkeo_variance" in features
    assert "EMG_submental_tkeo_max" in features
    assert "EOG_horizontal_permutation_entropy" not in features
    assert "EEG_Fpz-Cz_tkeo_mean" not in features


def test_spectral_features_use_four_second_welch_resolution():
    time = np.arange(3000) / 100.0
    x = np.sin(2 * np.pi * 1.0 * time)
    features = spectral_features(x, sample_rate_hz=100.0)
    assert features["delta_power"] > features["beta_power"]


def test_sequential_context_stays_inside_one_recording():
    X = pd.DataFrame({"EEG_Fpz-Cz_theta_relative_power": [0.1, 0.2, 0.3, 9.9]})
    metadata = pd.DataFrame(
        {
            "record_id": ["A", "A", "A", "B"],
            "epoch_index": [0, 1, 2, 0],
        }
    )
    expanded = add_sequential_context(X, metadata, lags=1)
    assert expanded.loc[1, "EEG_Fpz-Cz_theta_relative_power_lag1"] == 0.1
    assert expanded.loc[1, "EEG_Fpz-Cz_theta_relative_power_lead1"] == 0.3
    assert expanded.loc[3, "EEG_Fpz-Cz_theta_relative_power_lag1"] == 0.0


def test_hjorth_and_entropy_are_finite():
    x = np.sin(np.linspace(0, 2 * np.pi, 300))
    activity, mobility, complexity = hjorth_parameters(x)
    assert activity > 0
    assert mobility > 0
    assert complexity > 0
    assert 0 <= permutation_entropy(x) <= 1
