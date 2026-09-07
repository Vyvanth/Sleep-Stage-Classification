"""Temporal smoothing for sleep stage sequences."""

from __future__ import annotations

import numpy as np


def estimate_transition_matrix(labels: list[int] | np.ndarray, n_states: int, smoothing: float = 1.0) -> np.ndarray:
    """Estimate a smoothed transition matrix from sequential labels."""

    y = np.asarray(labels, dtype=int)
    counts = np.full((n_states, n_states), smoothing, dtype=float)
    for prev, cur in zip(y[:-1], y[1:]):
        counts[prev, cur] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi_smooth(
    probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    start_probabilities: np.ndarray | None = None,
    transition_weight: float = 0.35,
) -> np.ndarray:
    """Smooth epoch probabilities with Viterbi decoding."""

    probs = np.asarray(probabilities, dtype=float)
    trans = np.asarray(transition_matrix, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape (epochs, states).")
    n_steps, n_states = probs.shape
    if trans.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape must be states x states.")
    if transition_weight < 0:
        raise ValueError("transition_weight must be non-negative.")
    start = np.full(n_states, 1.0 / n_states) if start_probabilities is None else np.asarray(start_probabilities, dtype=float)
    eps = 1e-12
    log_probs = np.log(np.maximum(probs, eps))
    log_trans = transition_weight * np.log(np.maximum(trans, eps))
    log_start = np.log(np.maximum(start, eps))
    dp = np.zeros((n_steps, n_states), dtype=float)
    back = np.zeros((n_steps, n_states), dtype=int)
    dp[0] = log_start + log_probs[0]
    for t in range(1, n_steps):
        scores = dp[t - 1][:, None] + log_trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = scores[back[t], np.arange(n_states)] + log_probs[t]
    path = np.zeros(n_steps, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_steps - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def default_sleep_transition_matrix() -> np.ndarray:
    """A conservative AASM-like prior for W, N1, N2, N3, REM transitions."""

    matrix = np.array(
        [
            [0.82, 0.12, 0.03, 0.01, 0.02],
            [0.14, 0.20, 0.48, 0.04, 0.14],
            [0.04, 0.06, 0.70, 0.12, 0.08],
            [0.02, 0.02, 0.18, 0.76, 0.02],
            [0.12, 0.08, 0.18, 0.02, 0.60],
        ],
        dtype=float,
    )
    return matrix / matrix.sum(axis=1, keepdims=True)
