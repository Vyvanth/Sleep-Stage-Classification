import pandas as pd

from sleep_stage_classification.reporting import generate_grounded_report, sleep_metrics


def test_sleep_metrics_counts_sleep_epochs():
    predictions = pd.DataFrame({"predicted_stage": ["W", "W", "N1", "N2", "REM"]})
    metrics = sleep_metrics(predictions)
    assert metrics["total_recording_minutes"] == 2.5
    assert metrics["total_sleep_minutes"] == 1.5
    assert metrics["sleep_efficiency_percent"] == 60.0


def test_report_uses_attribution_evidence():
    predictions = pd.DataFrame({"predicted_stage": ["W", "N1", "N2"]})
    attributions = pd.DataFrame(
        [
            {"feature": "EEG_delta_power", "direction": "high", "attribution": 0.7},
            {"feature": "EMG_tkeo", "direction": "low", "attribution": 0.3},
        ]
    )
    report = generate_grounded_report(predictions, attributions)
    assert "Sleep Stage Classification Report" in report
    assert "EEG_delta_power" in report
