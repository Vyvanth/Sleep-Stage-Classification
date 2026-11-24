import pandas as pd

from sleep_stage_classification.gemini_reporting import _is_temporary_capacity_error, build_gemini_prompt


def test_gemini_prompt_contains_metrics_and_shap_evidence_without_record_id():
    predictions = pd.DataFrame(
        {
            "record_id": ["SC4001", "SC4001", "SC4001"],
            "predicted_stage": ["W", "N1", "N2"],
        }
    )
    attributions = pd.DataFrame(
        [{"feature": "EEG_delta_power", "direction": "increases", "attribution": 0.7}]
    )

    prompt = build_gemini_prompt(predictions, attributions)

    assert "EEG_delta_power" in prompt
    assert "SC4001" not in prompt
    assert "Do not diagnose" in prompt
    assert "Limitations and Required Review" in prompt


def test_temporary_capacity_error_detection():
    assert _is_temporary_capacity_error(RuntimeError("Error code: 503 - service_unavailable"))
    assert _is_temporary_capacity_error(RuntimeError("Gemini is experiencing high demand"))
    assert not _is_temporary_capacity_error(RuntimeError("Invalid API key"))
