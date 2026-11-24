import numpy as np

from sleep_stage_classification.preprocessing import (
    bandpass_filter,
    epoch_signal,
    filter_psg_recording,
    wavelet_denoise,
    zscore_per_channel,
)


def test_preprocessing_uses_30_second_100_hz_epochs_and_subject_channel_zscores():
    signal = np.vstack([np.arange(6000), np.arange(6000) * 2.0])
    normalized = zscore_per_channel(signal)
    epochs = epoch_signal(normalized, sample_rate_hz=100.0, epoch_seconds=30)

    assert epochs.shape == (2, 2, 3000)
    assert np.allclose(normalized.mean(axis=1), 0.0)
    assert np.allclose(normalized.std(axis=1), 1.0)


def test_wavelet_denoising_preserves_signal_shape():
    signal = np.random.default_rng(5).normal(size=(4, 3000))
    assert wavelet_denoise(signal, wavelet="db4", level=7).shape == signal.shape


def test_bandpass_filter_preserves_length_and_removes_dc():
    time = np.arange(3000) / 100.0
    signal = 4.0 + np.sin(2 * np.pi * 10.0 * time)
    filtered = bandpass_filter(signal, sample_rate_hz=100.0, low_hz=0.3, high_hz=35.0)
    assert filtered.shape == signal.shape
    assert abs(filtered.mean()) < abs(signal.mean())


def test_filter_psg_recording_skips_low_rate_emg():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(2, 1000))
    filtered = filter_psg_recording(
        signal,
        ["EEG Fpz-Cz", "EMG submental"],
        sample_rate_hz=100.0,
        native_rates={"EEG Fpz-Cz": 100.0, "EMG submental": 1.0},
    )
    assert np.allclose(filtered[1], signal[1])
    assert not np.allclose(filtered[0], signal[0])
