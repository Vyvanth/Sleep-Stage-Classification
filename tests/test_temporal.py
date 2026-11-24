import numpy as np
import pytest

from sleep_stage_classification.temporal import (
    estimate_transition_by_record,
    estimate_transition_matrix,
    smooth_by_record,
    viterbi_smooth,
)
from sleep_stage_classification.temporal_models import WindowBiLSTM


def test_transition_matrix_rows_sum_to_one():
    trans = estimate_transition_matrix([0, 0, 1, 1, 2], n_states=3)
    assert np.allclose(trans.sum(axis=1), 1)


def test_viterbi_prefers_temporally_consistent_path():
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.45, 0.55],
            [0.90, 0.10],
        ]
    )
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    path = viterbi_smooth(probabilities, transition, transition_weight=1.0)
    assert path.tolist() == [0, 0, 0]


def test_viterbi_accepts_class_logit_biases():
    probabilities = np.array([[0.55, 0.45], [0.55, 0.45]])
    transition = np.eye(2)

    path = viterbi_smooth(probabilities, transition, class_logit_biases=np.array([0.0, 0.5]))

    assert path.tolist() == [1, 1]


def test_record_aware_transition_estimation_does_not_join_record_boundaries():
    transition = estimate_transition_by_record(
        labels=np.array([0, 0, 1, 1]),
        record_ids=np.array(["first", "first", "second", "second"]),
        epoch_indices=np.array([0, 1, 0, 1]),
        n_states=2,
        smoothing=0.0,
    )

    assert np.allclose(transition, np.eye(2))


def test_record_aware_smoothing_preserves_input_row_order():
    probabilities = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    transition = np.eye(2)
    output = smooth_by_record(
        probabilities,
        record_ids=np.array(["second", "first", "second", "first"]),
        epoch_indices=np.array([1, 1, 0, 0]),
        transition_matrix=transition,
        transition_weight=0.0,
    )

    assert output.tolist() == [1, 0, 1, 0]


def test_bilstm_uses_batched_predictions_without_dropping_sequence_centres():
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(24, 3)).astype(np.float32)
    labels = np.tile(np.arange(5), 5)[:24]
    records = np.array(["a"] * 12 + ["b"] * 12)
    epochs = np.tile(np.arange(12), 2)
    model = WindowBiLSTM(feature_count=3, window_size=3, hidden_size=4, batch_size=4)

    model.fit(X, labels, records, epochs, epochs=1, patience=1)
    prediction = model.predict(X, records, epochs)

    assert np.all(prediction[[0, 11, 12, 23]] == -1)
    assert np.all(prediction[1:11] >= 0)
