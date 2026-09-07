import numpy as np

from sleep_stage_classification.evaluation import comparison_tables, confusion_matrix_frame


def test_comparison_tables_include_n1_for_each_model():
    y_true = np.array([0, 1, 1, 2])
    predictions = {
        "baseline_epoch_independent": np.array([0, 1, 2, 2]),
        "temporal_viterbi": np.array([0, 1, 1, 2]),
    }
    overall, per_class = comparison_tables(y_true, predictions, ["N1", "N2", "N3"])
    assert overall["model"].tolist() == ["baseline_epoch_independent", "temporal_viterbi"]
    assert set(per_class["stage"]) == {"N1", "N2", "N3"}


def test_confusion_matrix_frame_has_labels():
    matrix = confusion_matrix_frame(np.array([0, 1]), np.array([0, 0]), ["N1", "N2"])
    assert matrix.index.tolist() == ["true_N1", "true_N2"]
    assert matrix.columns.tolist() == ["pred_N1", "pred_N2"]
