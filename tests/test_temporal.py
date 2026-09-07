import numpy as np

from sleep_stage_classification.temporal import estimate_transition_matrix, viterbi_smooth


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
