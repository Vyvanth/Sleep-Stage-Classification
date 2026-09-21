"""Evaluation utilities for evidence-grounded LLM sleep reports."""

from __future__ import annotations

import re

import pandas as pd

from .reporting import build_evidence_summary, generate_grounded_report, sleep_metrics


def factual_consistency_report(
    generated_report: str,
    predictions: pd.DataFrame,
    attributions: pd.DataFrame,
) -> dict[str, object]:
    """Check that required numeric facts and SHAP feature names appear verbatim."""

    metrics = sleep_metrics(predictions)
    expected_metrics = {
        "total_recording_minutes": f"{metrics['total_recording_minutes']:.1f}",
        "total_sleep_minutes": f"{metrics['total_sleep_minutes']:.1f}",
        "sleep_efficiency_percent": f"{metrics['sleep_efficiency_percent']:.1f}",
        "sleep_latency_minutes": f"{metrics['sleep_latency_minutes']:.1f}",
        "wake_percent": f"{metrics['wake_percent']:.1f}",
        "n1_percent": f"{metrics['n1_percent']:.1f}",
        "n2_percent": f"{metrics['n2_percent']:.1f}",
        "n3_percent": f"{metrics['n3_percent']:.1f}",
        "rem_percent": f"{metrics['rem_percent']:.1f}",
    }
    found_metrics = [name for name, value in expected_metrics.items() if value in generated_report]
    expected_features = [item.split(" (mean attribution", 1)[0].split(" ", 1)[1] for item in build_evidence_summary(attributions)]
    found_features = [feature for feature in expected_features if feature in generated_report]
    metric_coverage = len(found_metrics) / len(expected_metrics) if expected_metrics else 1.0
    feature_coverage = len(found_features) / len(expected_features) if expected_features else 1.0
    limitation_present = bool(re.search(r"model prediction|clinical ground truth|not.*diagnos", generated_report, flags=re.IGNORECASE))
    return {
        "metric_fact_coverage": metric_coverage,
        "shap_feature_coverage": feature_coverage,
        "limitation_present": limitation_present,
        "factual_consistency_score": (metric_coverage + feature_coverage + float(limitation_present)) / 3,
        "missing_metric_facts": [name for name in expected_metrics if name not in found_metrics],
        "missing_shap_features": [feature for feature in expected_features if feature not in found_features],
    }


def bertscore_against_reference(generated_report: str, reference_report: str) -> dict[str, float]:
    """Calculate BERTScore against the deterministic evidence-grounded report."""

    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("Install bert-score to use semantic report evaluation.") from exc

    precision, recall, f1 = score(
        [generated_report],
        [reference_report],
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return {
        "bertscore_precision": float(precision[0]),
        "bertscore_recall": float(recall[0]),
        "bertscore_f1": float(f1[0]),
    }


def evaluate_report(
    generated_report: str,
    predictions: pd.DataFrame,
    attributions: pd.DataFrame,
    use_bertscore: bool = False,
) -> dict[str, object]:
    """Evaluate a Gemini report against its source facts and reference report."""

    result = factual_consistency_report(generated_report, predictions, attributions)
    if use_bertscore:
        reference_report = generate_grounded_report(predictions, attributions)
        result.update(bertscore_against_reference(generated_report, reference_report))
    return result
