"""Evidence-grounded sleep report generation."""

from __future__ import annotations

from collections import Counter

import pandas as pd


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
    return {
        "total_recording_minutes": total_minutes,
        "total_sleep_minutes": sleep_minutes,
        "sleep_efficiency_percent": 100.0 * sleep_minutes / total_minutes if total_minutes else 0.0,
        "sleep_latency_minutes": latency,
        "wake_percent": 100.0 * counts.get("W", 0) / total_epochs if total_epochs else 0.0,
        "n1_percent": 100.0 * counts.get("N1", 0) / total_epochs if total_epochs else 0.0,
        "n2_percent": 100.0 * counts.get("N2", 0) / total_epochs if total_epochs else 0.0,
        "n3_percent": 100.0 * counts.get("N3", 0) / total_epochs if total_epochs else 0.0,
        "rem_percent": 100.0 * counts.get("REM", 0) / total_epochs if total_epochs else 0.0,
    }


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
    """Generate a concise report grounded in predictions and attribution rows."""

    metrics = sleep_metrics(predictions)
    evidence = build_evidence_summary(attributions if attributions is not None else pd.DataFrame())
    lines = [
        "Sleep Stage Classification Report",
        "",
        "Overall summary",
        f"- Total recording time: {metrics['total_recording_minutes']:.1f} minutes",
        f"- Total sleep time: {metrics['total_sleep_minutes']:.1f} minutes",
        f"- Sleep efficiency: {metrics['sleep_efficiency_percent']:.1f}%",
        f"- Sleep latency: {metrics['sleep_latency_minutes']:.1f} minutes",
        "",
        "Sleep architecture",
        f"- Wake: {metrics['wake_percent']:.1f}%",
        f"- N1: {metrics['n1_percent']:.1f}%",
        f"- N2: {metrics['n2_percent']:.1f}%",
        f"- N3: {metrics['n3_percent']:.1f}%",
        f"- REM: {metrics['rem_percent']:.1f}%",
        "",
        "Physiological evidence used by the model",
    ]
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("- No attribution file was provided, so this report avoids feature-level claims.")
    lines.extend(
        [
            "",
            "Interpretation",
            "The report is generated from model outputs and feature evidence. It is intended for review by a qualified clinician or sleep researcher, not as a standalone diagnosis.",
        ]
    )
    return "\n".join(lines)
