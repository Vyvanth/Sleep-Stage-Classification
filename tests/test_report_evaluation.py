import pandas as pd

from sleep_stage_classification.report_evaluation import factual_consistency_report


def test_factual_consistency_reports_complete_coverage_for_matching_report():
    predictions = pd.DataFrame({"predicted_stage": ["W", "N1", "N2", "REM"]})
    attributions = pd.DataFrame(
        [{"feature": "EEG_delta_power", "direction": "increases", "attribution": 0.7}]
    )
    report = (
        "2.0 1.5 75.0 0.5 25.0 25.0 25.0 0.0 25.0 "
        "EEG_delta_power model predictions rather than clinical ground truth"
    )

    result = factual_consistency_report(report, predictions, attributions)

    assert result["factual_consistency_score"] == 1.0
