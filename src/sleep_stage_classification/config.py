"""Shared constants for sleep staging experiments."""

from __future__ import annotations

from dataclasses import dataclass


STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_ID = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
ID_TO_STAGE = {idx: stage for stage, idx in STAGE_TO_ID.items()}

RAW_STAGE_ALIASES = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "W": "W",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "N4": "N3",
    "R": "REM",
    "REM": "REM",
}

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

DEFAULT_MODALITY_WEIGHTS = {"EEG": 0.5, "EOG": 0.3, "EMG": 0.2}


@dataclass(frozen=True)
class PipelineConfig:
    """Parameters shared by preprocessing and feature extraction."""

    sample_rate_hz: float = 100.0
    epoch_seconds: int = 30
    selected_channels: tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
    random_state: int = 103
