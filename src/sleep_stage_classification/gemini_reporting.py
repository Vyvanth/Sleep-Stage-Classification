"""Gemini generation for reports constrained to model evidence."""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from .reporting import build_evidence_summary, sleep_metrics, stage_rows


def _is_temporary_capacity_error(error: Exception) -> bool:
    """Return whether Gemini reported a retryable capacity error."""

    message = str(error).lower()
    return "503" in message or "service_unavailable" in message or "high demand" in message


def build_gemini_prompt(predictions: pd.DataFrame, attributions: pd.DataFrame | None = None) -> str:
    """Build a de-identified, evidence-only prompt for a single recording."""

    metrics = sleep_metrics(predictions)
    evidence = build_evidence_summary(attributions if attributions is not None else pd.DataFrame())
    fact_sheet = {
        "model_derived_sleep_metrics": {key: round(value, 1) for key, value in metrics.items()},
        "model_predicted_stage_distribution": stage_rows(predictions),
        "shap_feature_evidence": evidence,
    }
    return (
        "Write a polished, clinically styled but strictly non-diagnostic automated sleep-stage analysis. "
        "It should resemble the organized structure of a polysomnogram report without impersonating a doctor.\n"
        "Rules:\n"
        "- Use only the supplied values and SHAP evidence.\n"
        "- Describe stages as model predictions, never as clinical ground truth.\n"
        "- Do not diagnose, provide treatment advice, call a result normal or abnormal, or invent thresholds, causes, symptoms, or feature effects.\n"
        "- Do not invent patient demographics, referral history, clinician names, signatures, AHI, oxygen data, arousals, cardiac findings, limb movements, or recommendations.\n"
        "- Mention that SHAP directions describe contributions to model scores, not physiological findings.\n"
        "- Use exactly these Markdown headings: Automated Sleep Stage Analysis; Recording and Method; Sleep Continuity; Sleep Architecture; Model Evidence; Automated Summary; Limitations and Required Review.\n"
        "- Present every numeric fact in a compact table or labelled line, retaining supplied rounding.\n"
        "- State clearly that this is an automated research-use output, not a diagnostic polysomnogram or physician interpretation.\n"
        "- Write 220 to 320 words.\n\n"
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

    prompt = build_gemini_prompt(predictions, attributions)
    for attempt in range(3):
        try:
            # A short-lived client avoids a Windows SDK retry bug that can
            # reuse a closed HTTP connection after a failed request.
            with genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=60_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ) as client:
                interaction = client.interactions.create(
                    model=model_name,
                    input=prompt,
                    generation_config={"thinking_level": "low", "temperature": 0.1},
                )
            break
        except Exception as exc:
            if not _is_temporary_capacity_error(exc) or attempt == 2:
                if _is_temporary_capacity_error(exc):
                    raise RuntimeError(
                        f"Gemini model {model_name!r} remained unavailable after three attempts. "
                        "Please try again later."
                    ) from exc
                raise
            delay_seconds = 2**attempt
            print(f"Gemini is temporarily busy. Retrying in {delay_seconds} seconds ({attempt + 1}/3)...")
            time.sleep(delay_seconds)
    report = getattr(interaction, "output_text", None)
    if not report:
        raise RuntimeError("Gemini returned no report text.")
    return report.strip()
