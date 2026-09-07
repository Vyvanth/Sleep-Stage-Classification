"""Signal preprocessing utilities for PSG recordings."""

from __future__ import annotations

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


def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4", level: int = 7) -> np.ndarray:
    """Denoise a signal with PyWavelets when available.

    If PyWavelets is not installed, this function returns the input unchanged.
    That fallback keeps the pipeline usable for smoke tests while making the
    optional production dependency explicit.
    """

    try:
        import pywt
    except ImportError:
        return np.asarray(signal, dtype=float)

    arr = np.asarray(signal, dtype=float)
    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=level, axis=-1)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    threshold = sigma * np.sqrt(2 * np.log(arr.shape[-1]))
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet=wavelet, axis=-1)
    return rec[..., : arr.shape[-1]]
