"""Gemini generation for reports constrained to model evidence."""

from __future__ import annotations

import json
import os

import pandas as pd

from .reporting import build_evidence_summary, sleep_metrics


def build_gemini_prompt(predictions: pd.DataFrame, attributions: pd.DataFrame | None = None) -> str:
    """Build a de-identified, evidence-only prompt for a single recording."""

    metrics = sleep_metrics(predictions)
    evidence = build_evidence_summary(attributions if attributions is not None else pd.DataFrame())
    fact_sheet = {
        "model_derived_sleep_metrics": {key: round(value, 1) for key, value in metrics.items()},
        "shap_feature_evidence": evidence,
    }
    return (
        "Write a concise research report from the fact sheet below.\n"
        "Rules:\n"
        "- Use only the supplied values and SHAP evidence.\n"
        "- Describe stages as model predictions, never as clinical ground truth.\n"
        "- Do not diagnose, provide treatment advice, call a result normal or abnormal, "
        "or invent thresholds, causes, symptoms, or feature effects.\n"
        "- Mention that SHAP directions describe contributions to model scores.\n"
        "- Use headings: Model Summary, Sleep Architecture, Model Evidence, Limitation.\n"
        "- Write all four requested sections, in 120 to 180 words total.\n\n"
        "Fact sheet:\n"
        f"{json.dumps(fact_sheet, indent=2)}"
    )


def generate_gemini_report(
    predictions: pd.DataFrame,
    attributions: pd.DataFrame | None = None,
    model_name: str = "gemini-3.6-flash",
) -> str:
    """Generate an evidence-bounded report through the Gemini API."""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY before generating a Gemini report.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("Install google-genai before generating a Gemini report.") from exc

    # A short-lived client avoids a Windows SDK retry bug that can reuse a
    # closed HTTP connection and conceal the actual Google API response.
    with genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        interaction = client.interactions.create(
            model=model_name,
            input=build_gemini_prompt(predictions, attributions),
            generation_config={"thinking_level": "low", "temperature": 0.1},
        )
    report = getattr(interaction, "output_text", None)
    if not report:
        raise RuntimeError("Gemini returned no report text.")
    return report.strip()
