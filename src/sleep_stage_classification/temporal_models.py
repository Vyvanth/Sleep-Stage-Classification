"""Optional learned temporal models for held-out sleep-stage evaluation."""

from __future__ import annotations

import numpy as np

from .config import PipelineConfig


def record_sequences(record_ids: np.ndarray, epoch_indices: np.ndarray) -> list[np.ndarray]:
    """Return chronologically ordered row positions for each recording."""

    return [
        positions[np.argsort(epoch_indices[positions])]
        for record in np.unique(record_ids)
        for positions in [np.flatnonzero(record_ids == record)]
    ]


class TorchLinearChainCRF:
    """Small linear-chain CRF using baseline probability vectors as emissions."""

    def __init__(self, n_states: int = 5):
        import torch

        self.torch = torch
        torch.manual_seed(PipelineConfig().random_state)
        self.n_states = n_states
        self.emission = torch.nn.Linear(n_states, n_states)
        self.transitions = torch.nn.Parameter(torch.zeros(n_states, n_states))
        self.start = torch.nn.Parameter(torch.zeros(n_states))
        self.end = torch.nn.Parameter(torch.zeros(n_states))

    def _log_partition(self, emissions):
        alpha = self.start + emissions[0]
        for emission in emissions[1:]:
            alpha = self.torch.logsumexp(alpha[:, None] + self.transitions, dim=0) + emission
        return self.torch.logsumexp(alpha + self.end, dim=0)

    def _path_score(self, emissions, labels):
        score = self.start[labels[0]] + emissions[0, labels[0]]
        for step in range(1, len(labels)):
            score = score + self.transitions[labels[step - 1], labels[step]] + emissions[step, labels[step]]
        return score + self.end[labels[-1]]

    def fit(self, probabilities, labels, record_ids, epoch_indices, epochs: int = 20, learning_rate: float = 0.01):
        torch = self.torch
        optimizer = torch.optim.Adam([*self.emission.parameters(), self.transitions, self.start, self.end], lr=learning_rate)
        sequences = record_sequences(np.asarray(record_ids), np.asarray(epoch_indices))
        inputs = torch.tensor(np.asarray(probabilities), dtype=torch.float32)
        targets = torch.tensor(np.asarray(labels), dtype=torch.long)
        self.emission.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for sequence in sequences:
                emissions = self.emission(inputs[sequence])
                loss = loss + self._log_partition(emissions) - self._path_score(emissions, targets[sequence])
            (loss / len(sequences)).backward()
            optimizer.step()
        return self

    def predict(self, probabilities, record_ids, epoch_indices):
        torch = self.torch
        output = np.zeros(len(probabilities), dtype=int)
        inputs = torch.tensor(np.asarray(probabilities), dtype=torch.float32)
        self.emission.eval()
        with torch.no_grad():
            for sequence in record_sequences(np.asarray(record_ids), np.asarray(epoch_indices)):
                emissions = self.emission(inputs[sequence]).numpy()
                dp = self.start.detach().numpy() + emissions[0]
                back = np.zeros((len(sequence), self.n_states), dtype=int)
                transitions = self.transitions.detach().numpy()
                for step in range(1, len(sequence)):
                    scores = dp[:, None] + transitions
                    back[step] = np.argmax(scores, axis=0)
                    dp = scores[back[step], np.arange(self.n_states)] + emissions[step]
                path = np.zeros(len(sequence), dtype=int)
                path[-1] = int(np.argmax(dp + self.end.detach().numpy()))
                for step in range(len(sequence) - 2, -1, -1):
                    path[step] = back[step + 1, path[step + 1]]
                output[sequence] = path
        return output


class WindowBiLSTM:
    """BiLSTM that predicts the centre label of an odd-length epoch window."""

    def __init__(self, feature_count: int, window_size: int = 11, hidden_size: int = 32):
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd value of at least 3.")
        import torch

        self.torch = torch
        torch.manual_seed(PipelineConfig().random_state)
        self.window_size = window_size
        self.encoder = torch.nn.LSTM(feature_count, hidden_size, batch_first=True, bidirectional=True)
        self.head = torch.nn.Linear(hidden_size * 2, 5)

    def _windows(self, X, labels, record_ids, epoch_indices):
        radius = self.window_size // 2
        windows, targets, positions = [], [], []
        for sequence in record_sequences(np.asarray(record_ids), np.asarray(epoch_indices)):
            for center in range(radius, len(sequence) - radius):
                window = sequence[center - radius : center + radius + 1]
                windows.append(X[window])
                targets.append(labels[sequence[center]])
                positions.append(sequence[center])
        return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=int), np.asarray(positions, dtype=int)

    def fit(self, X, labels, record_ids, epoch_indices, epochs: int = 15, patience: int = 3):
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import f1_score

        torch = self.torch
        X = np.asarray(X, dtype=np.float32)
        mean, std = X.mean(axis=0), X.std(axis=0)
        self.mean_, self.std_ = mean, np.maximum(std, 1e-8)
        X = (X - self.mean_) / self.std_
        windows, targets, positions = self._windows(X, np.asarray(labels), record_ids, epoch_indices)
        if len(windows) == 0:
            raise ValueError("No complete sequence windows were available for BiLSTM training.")
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=103)
        window_groups = np.asarray(record_ids)[positions]
        train_idx, validation_idx = next(split.split(windows, targets, window_groups))
        counts = np.bincount(targets[train_idx], minlength=5)
        weights = counts.sum() / np.maximum(counts, 1)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.Adam([*self.encoder.parameters(), *self.head.parameters()], lr=0.001)
        best_state, best_score, remaining = None, -1.0, patience
        for _ in range(epochs):
            self.encoder.train(); self.head.train(); optimizer.zero_grad()
            output, _ = self.encoder(torch.tensor(windows[train_idx]))
            loss = criterion(self.head(output[:, self.window_size // 2]), torch.tensor(targets[train_idx]))
            loss.backward(); optimizer.step()
            self.encoder.eval(); self.head.eval()
            with torch.no_grad():
                valid_output, _ = self.encoder(torch.tensor(windows[validation_idx]))
                prediction = self.head(valid_output[:, self.window_size // 2]).argmax(dim=1).numpy()
            score = f1_score(targets[validation_idx], prediction, average="macro", zero_division=0)
            if score > best_score:
                best_score, remaining = score, patience
                best_state = ([item.detach().clone() for item in self.encoder.parameters()], [item.detach().clone() for item in self.head.parameters()])
            else:
                remaining -= 1
                if remaining == 0:
                    break
        for parameter, value in zip(self.encoder.parameters(), best_state[0]): parameter.data.copy_(value)
        for parameter, value in zip(self.head.parameters(), best_state[1]): parameter.data.copy_(value)
        self.validation_macro_f1_ = best_score
        return self

    def predict(self, X, record_ids, epoch_indices):
        torch = self.torch
        normalized = (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_
        dummy = np.zeros(len(normalized), dtype=int)
        windows, _, positions = self._windows(normalized, dummy, record_ids, epoch_indices)
        output = np.full(len(normalized), -1, dtype=int)
        self.encoder.eval(); self.head.eval()
        with torch.no_grad():
            values, _ = self.encoder(torch.tensor(windows))
            output[positions] = self.head(values[:, self.window_size // 2]).argmax(dim=1).numpy()
        return output
