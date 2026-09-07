"""Feature engineering for multimodal PSG epochs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import DEFAULT_BANDS


def hjorth_parameters(x: np.ndarray) -> tuple[float, float]:
    """Return Hjorth mobility and complexity."""

    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0.0
    complexity = mobility_dx / mobility if mobility > 0 else 0.0
    return float(mobility), float(complexity)


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Compute normalized permutation entropy for one epoch."""

    x = np.asarray(x, dtype=float)
    n = len(x) - delay * (order - 1)
    if n <= 1:
        return 0.0
    patterns: dict[tuple[int, ...], int] = {}
    for i in range(n):
        window = x[i : i + delay * order : delay]
        key = tuple(np.argsort(window, kind="mergesort"))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(math.factorial(order))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def teager_kaiser_energy(x: np.ndarray) -> float:
    """Return mean Teager-Kaiser energy for transient activity."""

    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0
    energy = x[1:-1] ** 2 - x[:-2] * x[2:]
    return float(np.mean(np.abs(energy)))


def band_powers(x: np.ndarray, sample_rate_hz: float, bands: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    """Estimate absolute and relative FFT band powers."""

    bands = bands or DEFAULT_BANDS
    x = np.asarray(x, dtype=float)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    total = float(np.sum(spectrum[(freqs >= 0.5) & (freqs <= 30.0)]))
    out: dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        power = float(np.sum(spectrum[mask]))
        out[f"{name}_power"] = power
        out[f"{name}_relative_power"] = power / total if total > 0 else 0.0
    return out


def infer_modality(channel_name: str) -> str:
    upper = channel_name.upper()
    if "EOG" in upper:
        return "EOG"
    if "EMG" in upper:
        return "EMG"
    return "EEG"


def extract_epoch_features(epoch: np.ndarray, channel_names: list[str], sample_rate_hz: float) -> dict[str, float]:
    """Extract features for one epoch with shape channels x samples."""

    features: dict[str, float] = {}
    for channel_idx, channel_name in enumerate(channel_names):
        safe_name = channel_name.replace(" ", "_").replace("/", "_")
        x = np.asarray(epoch[channel_idx], dtype=float)
        prefix = f"{safe_name}"
        features[f"{prefix}_mean"] = float(np.mean(x))
        features[f"{prefix}_std"] = float(np.std(x))
        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(x**2)))
        features[f"{prefix}_ptp"] = float(np.ptp(x))
        mobility, complexity = hjorth_parameters(x)
        features[f"{prefix}_hjorth_mobility"] = mobility
        features[f"{prefix}_hjorth_complexity"] = complexity
        features[f"{prefix}_permutation_entropy"] = permutation_entropy(x)
        features[f"{prefix}_tkeo"] = teager_kaiser_energy(x)
        for band_name, value in band_powers(x, sample_rate_hz).items():
            features[f"{prefix}_{band_name}"] = value
    return features


def extract_feature_frame(
    epochs: np.ndarray,
    labels: list[str] | np.ndarray,
    channel_names: list[str],
    sample_rate_hz: float,
    subject_id: str,
    record_id: str,
) -> pd.DataFrame:
    """Build one feature row per epoch."""

    rows = []
    for idx, epoch in enumerate(epochs):
        row = {
            "subject_id": subject_id,
            "record_id": record_id,
            "epoch_index": idx,
            "stage": labels[idx],
        }
        row.update(extract_epoch_features(epoch, channel_names, sample_rate_hz))
        rows.append(row)
    return pd.DataFrame(rows)
