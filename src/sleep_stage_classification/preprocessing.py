"""Signal preprocessing utilities for PSG recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def zscore_per_channel(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply per-channel z-score normalization to a channel x sample array."""

    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected signal with shape (channels, samples).")
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return (arr - mean) / np.maximum(std, eps)


def epoch_signal(signal: np.ndarray, sample_rate_hz: float, epoch_seconds: int = 30) -> np.ndarray:
    """Split a channel x sample signal into non-overlapping 30-second epochs."""

    arr = np.asarray(signal, dtype=float)
    samples_per_epoch = int(round(sample_rate_hz * epoch_seconds))
    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive.")
    usable = (arr.shape[1] // samples_per_epoch) * samples_per_epoch
    if usable == 0:
        raise ValueError("Signal is shorter than one epoch.")
    trimmed = arr[:, :usable]
    epochs = trimmed.reshape(arr.shape[0], -1, samples_per_epoch)
    return np.moveaxis(epochs, 1, 0)


def bandpass_filter(
    signal: np.ndarray,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along the last axis."""

    from scipy.signal import butter, sosfiltfilt

    arr = np.asarray(signal, dtype=float)
    nyquist = sample_rate_hz / 2.0
    high = min(float(high_hz), nyquist * 0.99)
    low = max(float(low_hz), 1e-3)
    if low >= high:
        return arr
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, arr, axis=-1)


def filter_psg_recording(
    signal: np.ndarray,
    channel_names: list[str],
    sample_rate_hz: float,
    native_rates: dict[str, float] | None = None,
) -> np.ndarray:
    """Apply AASM-style band limits, skipping 1 Hz cassette EMG envelopes."""

    native_rates = native_rates or {}
    filtered = np.asarray(signal, dtype=float).copy()
    for index, name in enumerate(channel_names):
        native_hz = float(native_rates.get(name, sample_rate_hz))
        upper = name.upper()
        if "EMG" in upper and native_hz < 20:
            continue
        if "EMG" in upper:
            low_hz, high_hz = 10.0, 45.0
        else:
            low_hz, high_hz = 0.3, 35.0
        filtered[index] = bandpass_filter(filtered[index], sample_rate_hz, low_hz, high_hz)
    return filtered


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 7,
    finest_levels: int = 2,
    threshold_scale: float = 0.5,
) -> np.ndarray:
    """Denoise a signal with PyWavelets when available.

    Only the finest detail bands are soft-thresholded, and the universal
    threshold is scaled down so sleep spindles and K-complexes are retained.
    If PyWavelets is not installed, this function returns the input unchanged.
    """

    try:
        import pywt
    except ImportError:
        return np.asarray(signal, dtype=float)

    arr = np.asarray(signal, dtype=float)
    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=level, axis=-1)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    threshold = threshold_scale * sigma * np.sqrt(2 * np.log(max(arr.shape[-1], 2)))
    start = max(len(coeffs) - finest_levels, 1)
    for index in range(start, len(coeffs)):
        coeffs[index] = pywt.threshold(coeffs[index], threshold, mode="soft")
    rec = pywt.waverec(coeffs, wavelet=wavelet, axis=-1)
    return rec[..., : arr.shape[-1]]


def plot_denoising_comparison(
    original: np.ndarray,
    denoised: np.ndarray,
    channel_names: list[str],
    sample_rate_hz: float,
    output_path: str | Path,
    seconds: int = 30,
) -> Path:
    """Save a before/after denoising plot for a short recording segment."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to create denoising plots.") from exc

    original = np.asarray(original, dtype=float)
    denoised = np.asarray(denoised, dtype=float)
    samples = min(original.shape[1], int(round(seconds * sample_rate_hz)))
    time = np.arange(samples) / sample_rate_hz
    figure, axes = plt.subplots(len(channel_names), 1, figsize=(12, 2.4 * len(channel_names)), sharex=True)
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(time, original[index, :samples], color="0.55", linewidth=0.7, label="Original")
        axis.plot(time, denoised[index, :samples], color="#1769aa", linewidth=0.8, label="Wavelet denoised")
        axis.set_ylabel(channel_names[index])
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle("Wavelet denoising quality check")
    figure.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return out
