"""Feature engineering for multimodal PSG epochs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import welch

from .config import DEFAULT_BANDS


def hjorth_parameters(x: np.ndarray) -> tuple[float, float, float]:
    """Return Hjorth activity, mobility, and complexity."""

    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0.0
    complexity = mobility_dx / mobility if mobility > 0 else 0.0
    return float(activity), float(mobility), float(complexity)


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Compute normalized permutation entropy for one epoch."""

    x = np.asarray(x, dtype=float)
    n = len(x) - delay * (order - 1)
    if n <= 1:
        return 0.0
    indices = np.arange(n)[:, np.newaxis] + delay * np.arange(order)
    patterns = np.argsort(x[indices], axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    counts = counts.astype(float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(math.factorial(order))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def tkeo_statistics(x: np.ndarray) -> dict[str, float]:
    """Return mean, maximum, and variance of Teager-Kaiser energy."""

    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return {"tkeo_mean": 0.0, "tkeo_max": 0.0, "tkeo_variance": 0.0}
    energy = x[1:-1] ** 2 - x[:-2] * x[2:]
    energy = np.abs(energy)
    return {
        "tkeo_mean": float(np.mean(energy)),
        "tkeo_max": float(np.max(energy)),
        "tkeo_variance": float(np.var(energy)),
    }


def spectral_features(x: np.ndarray, sample_rate_hz: float, bands: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    """Estimate Welch-PSD band powers, entropy, and physiologic band ratios."""

    bands = bands or DEFAULT_BANDS
    x = np.asarray(x, dtype=float)
    nperseg = min(max(int(round(sample_rate_hz * 4.0)), 8), len(x))
    freqs, psd = welch(
        x,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hamming",
    )
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total = float(np.trapezoid(psd[total_mask], freqs[total_mask])) if np.any(total_mask) else 0.0
    out: dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        power = float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        out[f"{name}_power"] = power
        out[f"{name}_relative_power"] = power / total if total > 0 else 0.0
    psd_values = psd[total_mask]
    if total > 0 and len(psd_values) > 1:
        probabilities = psd_values / psd_values.sum()
        nonzero = probabilities[probabilities > 0]
        out["spectral_entropy"] = float(-np.sum(nonzero * np.log2(nonzero)) / np.log2(len(probabilities)))
    else:
        out["spectral_entropy"] = 0.0
    for numerator, denominator in (("delta", "theta"), ("theta", "alpha"), ("alpha", "beta")):
        denominator_power = out[f"{denominator}_power"]
        out[f"{numerator}_{denominator}_power_ratio"] = (
            out[f"{numerator}_power"] / denominator_power if denominator_power > 0 else 0.0
        )
    return out


def time_domain_features(x: np.ndarray) -> dict[str, float]:
    """Return the requested time-domain features for one epoch."""

    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x)
    std = float(np.std(x))
    if std == 0:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        normalized = centered / std
        skewness = float(np.mean(normalized**3))
        excess_kurtosis = float(np.mean(normalized**4) - 3.0)
    zero_crossing_rate = float(np.mean(np.diff(np.signbit(centered)) != 0)) if len(x) > 1 else 0.0
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x)),
        "std": std,
        "rms": float(np.sqrt(np.mean(x**2))),
        "skewness": skewness,
        "kurtosis": excess_kurtosis,
        "zero_crossing_rate": zero_crossing_rate,
        "peak_to_peak": float(np.ptp(x)),
    }


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
        for feature_name, value in time_domain_features(x).items():
            features[f"{prefix}_{feature_name}"] = value
        activity, mobility, complexity = hjorth_parameters(x)
        features[f"{prefix}_hjorth_activity"] = activity
        features[f"{prefix}_hjorth_mobility"] = mobility
        features[f"{prefix}_hjorth_complexity"] = complexity
        for feature_name, value in spectral_features(x, sample_rate_hz).items():
            features[f"{prefix}_{feature_name}"] = value
        if "EEG" in channel_name.upper():
            features[f"{prefix}_permutation_entropy"] = permutation_entropy(x)
        if "EOG" in channel_name.upper() or "EMG" in channel_name.upper():
            for feature_name, value in tkeo_statistics(x).items():
                features[f"{prefix}_{feature_name}"] = value
    return features


CONTEXT_FEATURE_KEYS = (
    "relative_power",
    "power_ratio",
    "spectral_entropy",
    "permutation_entropy",
    "hjorth_mobility",
    "tkeo_mean",
)


def sequential_context_columns(columns: list[str]) -> list[str]:
    """Return the stage-discriminative columns that benefit from lag/lead copies."""

    return [column for column in columns if any(key in column for key in CONTEXT_FEATURE_KEYS)]


def add_sequential_context(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    lags: int = 1,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Copy neighbouring-epoch values from the same recording only.

    Sleep stages persist across epochs, so the previous and next theta/alpha
    ratio, EOG energy, and similar cues are legitimate Phase-1 features. Values
    are never taken from a different recording.
    """

    if lags <= 0:
        return X.copy()
    if len(X) != len(metadata):
        raise ValueError("Feature rows and metadata rows must be aligned.")
    selected = columns or sequential_context_columns(list(X.columns))
    if not selected:
        return X.copy()
    expanded = X.copy()
    record_ids = metadata["record_id"].to_numpy()
    epoch_indices = metadata["epoch_index"].to_numpy()
    for record_id in pd.unique(record_ids):
        positions = np.flatnonzero(record_ids == record_id)
        ordered = positions[np.argsort(epoch_indices[positions])]
        block = X.iloc[ordered][selected]
        for lag in range(1, lags + 1):
            previous = block.shift(lag)
            following = block.shift(-lag)
            expanded.loc[X.index[ordered], [f"{name}_lag{lag}" for name in selected]] = previous.to_numpy()
            expanded.loc[X.index[ordered], [f"{name}_lead{lag}" for name in selected]] = following.to_numpy()
    return expanded.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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
