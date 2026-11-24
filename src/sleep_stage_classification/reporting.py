"""Evidence-grounded sleep report generation."""

from __future__ import annotations

from collections import Counter

import pandas as pd


STAGE_LABELS = {
    "W": "Wake",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "REM": "REM",
}


def sleep_metrics(predictions: pd.DataFrame, epoch_seconds: int = 30) -> dict[str, float]:
    """Compute core sleep architecture metrics from predicted stages."""

    stages = predictions["predicted_stage"].tolist()
    counts = Counter(stages)
    total_epochs = len(stages)
    sleep_epochs = total_epochs - counts.get("W", 0)
    total_minutes = total_epochs * epoch_seconds / 60.0
    sleep_minutes = sleep_epochs * epoch_seconds / 60.0
    first_sleep = next((idx for idx, stage in enumerate(stages) if stage != "W"), None)
    latency = (first_sleep or 0) * epoch_seconds / 60.0 if first_sleep is not None else total_minutes
    wake_after_sleep_onset_epochs = sum(stage == "W" for stage in stages[(first_sleep or 0) :]) if first_sleep is not None else 0
    return {
        "total_recording_minutes": total_minutes,
        "total_sleep_minutes": sleep_minutes,
        "sleep_efficiency_percent": 100.0 * sleep_minutes / total_minutes if total_minutes else 0.0,
        "sleep_latency_minutes": latency,
        "wake_after_sleep_onset_minutes": wake_after_sleep_onset_epochs * epoch_seconds / 60.0,
        "wake_percent": 100.0 * counts.get("W", 0) / total_epochs if total_epochs else 0.0,
        "n1_percent": 100.0 * counts.get("N1", 0) / total_epochs if total_epochs else 0.0,
        "n2_percent": 100.0 * counts.get("N2", 0) / total_epochs if total_epochs else 0.0,
        "n3_percent": 100.0 * counts.get("N3", 0) / total_epochs if total_epochs else 0.0,
        "rem_percent": 100.0 * counts.get("REM", 0) / total_epochs if total_epochs else 0.0,
    }


def stage_rows(predictions: pd.DataFrame, epoch_seconds: int = 30) -> list[dict[str, float | str]]:
    """Return sleep-stage durations with recording and model-predicted sleep percentages."""

    stages = predictions["predicted_stage"].tolist()
    counts = Counter(stages)
    total_epochs = len(stages)
    sleep_epochs = total_epochs - counts.get("W", 0)
    rows: list[dict[str, float | str]] = []
    for code in ("W", "N1", "N2", "N3", "REM"):
        count = counts.get(code, 0)
        rows.append(
            {
                "stage": STAGE_LABELS[code],
                "minutes": count * epoch_seconds / 60.0,
                "recording_percent": 100.0 * count / total_epochs if total_epochs else 0.0,
                "sleep_percent": 100.0 * count / sleep_epochs if code != "W" and sleep_epochs else 0.0,
            }
        )
    return rows


def build_evidence_summary(attributions: pd.DataFrame, top_k: int = 8) -> list[str]:
    """Summarize the most common attribution features."""

    if attributions.empty:
        return []
    grouped = (
        attributions.groupby(["feature", "direction"], as_index=False)["attribution"]
        .mean()
        .sort_values("attribution", ascending=False)
        .head(top_k)
    )
    return [f"{row.direction} {row.feature} (mean attribution {row.attribution:.3f})" for row in grouped.itertuples()]


def generate_grounded_report(predictions: pd.DataFrame, attributions: pd.DataFrame | None = None) -> str:
    """Generate a structured, non-diagnostic sleep-stage analysis report."""

    metrics = sleep_metrics(predictions)
    stages = stage_rows(predictions)
    evidence = build_evidence_summary(attributions if attributions is not None else pd.DataFrame())
    record_id = "Not supplied"
    if "record_id" in predictions and predictions["record_id"].nunique() == 1:
        record_id = str(predictions["record_id"].iloc[0])
    lines = [
        "AUTOMATED SLEEP STAGE ANALYSIS",
        "Research-use sleep architecture summary",
        "=" * 52,
        "",
        "REPORT DETAILS",
        f"Recording identifier: {record_id}",
        "Report status: Automated algorithm output; not a physician interpretation.",
        "Scoring basis: Model-predicted 30-second sleep-stage epochs from available EEG, EOG, and EMG features.",
        "",
        "SLEEP CONTINUITY",
        f"Total recording time: {metrics['total_recording_minutes']:.1f} minutes",
        f"Model-predicted total sleep time: {metrics['total_sleep_minutes']:.1f} minutes",
        f"Model-predicted sleep efficiency: {metrics['sleep_efficiency_percent']:.1f}%",
        f"Model-predicted sleep latency: {metrics['sleep_latency_minutes']:.1f} minutes",
        f"Model-predicted wake after sleep onset: {metrics['wake_after_sleep_onset_minutes']:.1f} minutes",
        "",
        "SLEEP ARCHITECTURE",
        "Stage   Duration (min)   % recording   % model-predicted sleep",
    ]
    lines.extend(
        f"{row['stage']:<5}   {row['minutes']:>12.1f}   {row['recording_percent']:>11.1f}%   {row['sleep_percent']:>22.1f}%"
        for row in stages
    )
    lines.extend(
        [
            "",
            "MODEL EVIDENCE",
            "The following feature names are an audit trail. SHAP direction indicates a contribution to model scores, not a physiological finding.",
        ]
    )
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("- No attribution file was provided, so this report avoids feature-level claims.")
    lines.extend(
        [
            "",
            "AUTOMATED SUMMARY",
            "This recording has been summarized from automated sleep-stage predictions. The architecture values above describe the model output and require review against the scored signal data.",
            "",
            "LIMITATIONS AND REQUIRED REVIEW",
            "This is not a diagnostic polysomnogram report and is not clinical ground truth. It does not assess respiratory events, oxygen saturation, arousals, cardiac rhythm, limb movements, symptoms, medications, or treatment needs. Review by a qualified sleep clinician or researcher is required before any clinical use.",
        ]
    )
    return "\n".join(lines)
