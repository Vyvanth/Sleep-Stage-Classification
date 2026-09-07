"""Dataset loading helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_STAGE_ALIASES, STAGE_ORDER
from .features import extract_epoch_features
from .preprocessing import wavelet_denoise, zscore_per_channel


METADATA_COLUMNS = {"subject_id", "record_id", "epoch_index", "stage"}


@dataclass(frozen=True)
class SleepEdfPair:
    """One PSG file and its matching hypnogram annotation file."""

    record_id: str
    subject_id: str
    psg_path: Path
    hypnogram_path: Path


def normalize_stage(raw_stage: str) -> str | None:
    """Map Sleep-EDF/ISRUC labels to W, N1, N2, N3, REM."""

    stage = RAW_STAGE_ALIASES.get(str(raw_stage).strip())
    return stage if stage in STAGE_ORDER else None


def sleep_edf_pair_key(path: str | Path) -> str:
    """Return the six-character Sleep-EDF recording key, e.g. SC4001."""

    name = Path(path).name
    match = re.match(r"([A-Z]{2}\d{4})", name)
    if not match:
        raise ValueError(f"Cannot infer Sleep-EDF key from {name}")
    return match.group(1)


def discover_sleep_edf_pairs(root: str | Path) -> list[SleepEdfPair]:
    """Find PSG/Hypnogram pairs below a Sleep-EDF Expanded directory."""

    root = Path(root)
    psg_files = sorted(root.rglob("*-PSG.edf"))
    hyp_files = {sleep_edf_pair_key(path): path for path in root.rglob("*-Hypnogram.edf")}
    pairs: list[SleepEdfPair] = []
    for psg_path in psg_files:
        key = sleep_edf_pair_key(psg_path)
        hyp_path = hyp_files.get(key)
        if hyp_path is None:
            continue
        pairs.append(
            SleepEdfPair(
                record_id=key,
                subject_id=key[:5],
                psg_path=psg_path,
                hypnogram_path=hyp_path,
            )
        )
    return pairs


def load_feature_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load a feature CSV and return X, y, metadata."""

    df = pd.read_csv(path)
    missing = METADATA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["stage"] = df["stage"].map(lambda value: normalize_stage(value) or value)
    feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["stage"]
    metadata = df[["subject_id", "record_id", "epoch_index"]]
    return X, y, metadata


def load_sleep_edf_record(psg_path: str | Path, hypnogram_path: str | Path, channels: list[str]):
    """Load one Sleep-EDF PSG/hypnogram pair with MNE.

    This helper intentionally stays thin because local channel naming differs
    between Sleep-EDF subsets. Downstream code can pass the channel list chosen
    for the experiment.
    """

    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install the 'full' extra to load EDF files with MNE.") from exc

    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    annotations = mne.read_annotations(str(hypnogram_path))
    raw.set_annotations(annotations)
    raw.pick_channels(channels)
    return raw


def _pick_available_channels(raw, requested_channels: list[str]) -> list[str]:
    available = {name.lower(): name for name in raw.ch_names}
    picked = []
    for requested in requested_channels:
        exact = available.get(requested.lower())
        if exact:
            picked.append(exact)
            continue
        compact_requested = re.sub(r"[^a-z0-9]+", "", requested.lower())
        for channel in raw.ch_names:
            compact_channel = re.sub(r"[^a-z0-9]+", "", channel.lower())
            if compact_requested in compact_channel or compact_channel in compact_requested:
                picked.append(channel)
                break
    return list(dict.fromkeys(picked))


def extract_sleep_edf_features(
    pair: SleepEdfPair,
    channels: list[str],
    sample_rate_hz: float = 100.0,
    epoch_seconds: int = 30,
    trim_wake_minutes: int = 30,
    denoise: bool = True,
) -> pd.DataFrame:
    """Load one Sleep-EDF pair and extract one feature row per scored epoch."""

    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install MNE first: pip install mne") from exc

    raw = mne.io.read_raw_edf(str(pair.psg_path), preload=True, verbose=False)
    annotations = mne.read_annotations(str(pair.hypnogram_path))
    raw.set_annotations(annotations)
    picked_channels = _pick_available_channels(raw, channels)
    if not picked_channels:
        raise ValueError(f"No requested channels found in {pair.psg_path.name}. Available: {raw.ch_names}")
    raw.pick(picked_channels)
    if abs(raw.info["sfreq"] - sample_rate_hz) > 1e-6:
        raw.resample(sample_rate_hz, npad="auto", verbose=False)

    rows = []
    samples_per_epoch = int(round(sample_rate_hz * epoch_seconds))
    for annotation in raw.annotations:
        stage = normalize_stage(annotation["description"])
        if stage is None:
            continue
        duration_epochs = int(round(float(annotation["duration"]) / epoch_seconds))
        if duration_epochs <= 0:
            continue
        for offset in range(duration_epochs):
            onset = float(annotation["onset"]) + offset * epoch_seconds
            start = raw.time_as_index(onset, use_rounding=True)[0]
            stop = start + samples_per_epoch
            if start < 0 or stop > raw.n_times:
                continue
            signal = raw.get_data(start=start, stop=stop)
            if signal.shape[1] != samples_per_epoch:
                continue
            if denoise:
                signal = wavelet_denoise(signal)
            signal = zscore_per_channel(signal)
            row = {
                "subject_id": pair.subject_id,
                "record_id": pair.record_id,
                "epoch_index": len(rows),
                "stage": stage,
            }
            row.update(extract_epoch_features(signal, picked_channels, sample_rate_hz))
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty or trim_wake_minutes <= 0:
        return frame
    return trim_outer_wake_epochs(frame, keep_minutes=trim_wake_minutes, epoch_seconds=epoch_seconds)


def trim_outer_wake_epochs(frame: pd.DataFrame, keep_minutes: int = 30, epoch_seconds: int = 30) -> pd.DataFrame:
    """Keep only a wake margin before first sleep and after last sleep."""

    sleep_mask = frame["stage"] != "W"
    if not sleep_mask.any():
        return frame
    margin_epochs = int(round(keep_minutes * 60 / epoch_seconds))
    sleep_indices = np.flatnonzero(sleep_mask.to_numpy())
    start = max(int(sleep_indices[0]) - margin_epochs, 0)
    stop = min(int(sleep_indices[-1]) + margin_epochs + 1, len(frame))
    trimmed = frame.iloc[start:stop].copy()
    trimmed["epoch_index"] = np.arange(len(trimmed))
    return trimmed


def build_sleep_edf_feature_csv(
    root: str | Path,
    output_path: str | Path,
    channels: list[str],
    max_records: int | None = None,
    sample_rate_hz: float = 100.0,
    trim_wake_minutes: int = 30,
    denoise: bool = True,
) -> Path:
    """Extract Sleep-EDF features from several records and write one CSV."""

    pairs = discover_sleep_edf_pairs(root)
    if max_records is not None:
        pairs = pairs[:max_records]
    if not pairs:
        raise ValueError(f"No Sleep-EDF PSG/Hypnogram pairs found below {root}")

    frames = []
    for idx, pair in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Extracting {pair.record_id} from {pair.psg_path.name}")
        frame = extract_sleep_edf_features(
            pair,
            channels=channels,
            sample_rate_hz=sample_rate_hz,
            trim_wake_minutes=trim_wake_minutes,
            denoise=denoise,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No valid scored epochs were extracted.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return out


def make_synthetic_feature_csv(path: str | Path, subjects: int = 6, epochs_per_subject: int = 90, random_state: int = 103) -> Path:
    """Create a deterministic PSG-like feature table for smoke tests."""

    rng = np.random.default_rng(random_state)
    rows = []
    transition = np.array(
        [
            [0.86, 0.08, 0.03, 0.01, 0.02],
            [0.12, 0.18, 0.54, 0.04, 0.12],
            [0.03, 0.05, 0.72, 0.10, 0.10],
            [0.02, 0.02, 0.18, 0.74, 0.04],
            [0.10, 0.08, 0.20, 0.02, 0.60],
        ]
    )
    stage_means = {
        "W": [1.3, 0.4, 0.3, 1.2],
        "N1": [0.7, 0.9, 0.5, 0.6],
        "N2": [0.4, 1.1, 0.9, 0.4],
        "N3": [0.2, 1.8, 0.5, 0.2],
        "REM": [0.5, 0.7, 1.5, 0.1],
    }
    names = ["alpha_relative_power", "delta_relative_power", "eog_tkeo", "emg_tkeo"]
    for subject in range(subjects):
        state = 0
        for epoch_idx in range(epochs_per_subject):
            if epoch_idx > 0:
                state = int(rng.choice(len(STAGE_ORDER), p=transition[state]))
            stage = STAGE_ORDER[state]
            values = np.asarray(stage_means[stage]) + rng.normal(0, 0.18, len(names))
            row = {
                "subject_id": f"S{subject:03d}",
                "record_id": f"S{subject:03d}_N1",
                "epoch_index": epoch_idx,
                "stage": stage,
            }
            row.update({name: max(float(value), 0.0) for name, value in zip(names, values)})
            rows.append(row)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
