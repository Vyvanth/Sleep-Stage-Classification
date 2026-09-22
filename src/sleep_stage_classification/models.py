"""Baseline classifier training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DEFAULT_MODALITY_WEIGHTS


@dataclass
class TrainingResult:
    model: Pipeline
    label_encoder: LabelEncoder
    feature_names: list[str]
    metrics: dict[str, float]
    split_strategy: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    classifier_name: str
    sampler_name: str
    feature_selection_name: str
    selected_feature_names: list[str]


class EnsembleFeatureSelector(BaseEstimator, TransformerMixin):
    """Five-fold, multimethod feature selection with modality and redundancy control.

    It combines LightGBM gain, mutual information, and permutation importance.
    Every score is calculated from training data only because this transformer is
    fitted inside the model pipeline after the subject-wise holdout is created.
    """

    def __init__(
        self,
        select_k: int = 67,
        preselect_k: int = 80,
        correlation_threshold: float = 0.9,
        n_splits: int = 5,
        permutation_repeats: int = 3,
        random_state: int = 103,
    ):
        self.select_k = select_k
        self.preselect_k = preselect_k
        self.correlation_threshold = correlation_threshold
        self.n_splits = n_splits
        self.permutation_repeats = permutation_repeats
        self.random_state = random_state

    def fit(self, X, y):
        X_frame = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(X_frame.columns, dtype=object)
        if X_frame.shape[1] == 0:
            raise ValueError("Feature selection requires at least one feature.")
        y_array = np.asarray(y)
        min_class_count = int(pd.Series(y_array).value_counts().min())
        folds = min(self.n_splits, min_class_count)
        if folds < 2:
            raise ValueError("Ensemble feature selection requires at least two samples in each class.")

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        aggregate_scores = np.zeros(X_frame.shape[1], dtype=float)
        stable_top_counts = np.zeros(X_frame.shape[1], dtype=float)
        # The paper retains the top 80 of 99 candidates before redundancy
        # filtering. Scale that proportion for smaller feature sets.
        fold_top_k = min(self.preselect_k, max(1, round(X_frame.shape[1] * 80 / 99)))
        for fold_index, (train_idx, validation_idx) in enumerate(splitter.split(X_frame, y_array)):
            train_X = X_frame.iloc[train_idx]
            validation_X = X_frame.iloc[validation_idx]
            train_y = y_array[train_idx]
            validation_y = y_array[validation_idx]
            scaler = StandardScaler()
            scored_train_X = pd.DataFrame(
                scaler.fit_transform(train_X), columns=X_frame.columns, index=train_X.index
            )
            scored_validation_X = pd.DataFrame(
                scaler.transform(validation_X), columns=X_frame.columns, index=validation_X.index
            )
            model = self._importance_model(fold_index)
            model.fit(scored_train_X, train_y)
            gain = self._gain_importance(model, X_frame.shape[1])
            mutual_information = mutual_info_classif(scored_train_X, train_y, random_state=self.random_state + fold_index)
            permutation = permutation_importance(
                model,
                scored_validation_X,
                validation_y,
                scoring="f1_macro",
                n_repeats=self.permutation_repeats,
                random_state=self.random_state + fold_index,
                n_jobs=1,
            ).importances_mean
            fold_score = (
                self._normalize_scores(gain)
                + self._normalize_scores(mutual_information)
                + self._normalize_scores(permutation)
            ) / 3.0
            aggregate_scores += fold_score
            stable_top_counts[np.argsort(fold_score)[::-1][:fold_top_k]] += 1

        stability = stable_top_counts / folds
        base_scores = aggregate_scores / folds
        modality_weights = np.asarray([self._modality_weight(name) for name in self.feature_names_in_])
        self.feature_scores_ = base_scores * stability * modality_weights
        ranked_indices = np.argsort(self.feature_scores_)[::-1][: min(self.preselect_k, X_frame.shape[1])]
        self.support_ = self._correlation_filtered_support(X_frame, ranked_indices)
        self.selected_feature_names_ = self.feature_names_in_[self.support_].tolist()
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.flatnonzero(self.support_) if indices else self.support_

    def _importance_model(self, fold_index: int):
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="multiclass",
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                colsample_bytree=0.9,
                random_state=self.random_state + fold_index,
                verbosity=-1,
            )
        except ImportError:
            return ExtraTreesClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=self.random_state + fold_index,
                n_jobs=1,
            )

    @staticmethod
    def _gain_importance(model, feature_count: int) -> np.ndarray:
        if hasattr(model, "booster_"):
            return np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
        return np.asarray(getattr(model, "feature_importances_", np.zeros(feature_count)), dtype=float)

    @staticmethod
    def _normalize_scores(values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        minimum, maximum = values.min(), values.max()
        return (values - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(values)

    @staticmethod
    def _modality_weight(feature_name: str) -> float:
        upper = str(feature_name).upper()
        if "EOG" in upper:
            return DEFAULT_MODALITY_WEIGHTS["EOG"]
        if "EMG" in upper:
            return DEFAULT_MODALITY_WEIGHTS["EMG"]
        return DEFAULT_MODALITY_WEIGHTS["EEG"]

    def _correlation_filtered_support(self, X: pd.DataFrame, ranked_indices: np.ndarray) -> np.ndarray:
        correlations = X.corr().abs().fillna(0.0).to_numpy()
        selected: list[int] = []
        for index in ranked_indices:
            if all(correlations[index, prior] < self.correlation_threshold for prior in selected):
                selected.append(int(index))
            if len(selected) >= min(self.select_k, X.shape[1]):
                break
        support = np.zeros(X.shape[1], dtype=bool)
        support[selected] = True
        return support


def make_baseline_classifier(classifier: str = "auto", random_state: int = 103):
    """Create the requested tabular classifier."""

    if classifier in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier

            return (
                LGBMClassifier(
                    objective="multiclass",
                    n_estimators=300,
                    learning_rate=0.04,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=random_state,
                    verbosity=-1,
                ),
                "lightgbm",
            )
        except ImportError:
            if classifier == "lightgbm":
                raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

    if classifier not in {"auto", "random_forest"}:
        raise ValueError("classifier must be one of: auto, lightgbm, random_forest")
    return (
        RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "random_forest",
    )


def make_feature_selector(feature_selection: str = "ensemble", select_k: int = 67, random_state: int = 103):
    """Create the supervised feature-selection stage used during training."""

    if feature_selection == "none":
        return None, "none"
    if feature_selection == "ensemble":
        if select_k <= 0:
            raise ValueError("select_k must be positive.")
        return EnsembleFeatureSelector(select_k=select_k, random_state=random_state), "ensemble_lgbm_mi_permutation"
    raise ValueError("feature_selection must be one of: none, ensemble")


def make_pipeline(
    classifier: str = "auto",
    sampler: str = "none",
    feature_selection: str = "ensemble",
    select_k: int = 67,
    random_state: int = 103,
):
    """Create a training pipeline with optional imbalance handling."""

    estimator, classifier_name = make_baseline_classifier(classifier=classifier, random_state=random_state)
    selector, selector_name = make_feature_selector(feature_selection, select_k, random_state)
    if sampler == "none":
        steps = []
        if selector is not None:
            steps.append(("select", selector))
        steps.append(("scale", StandardScaler()))
        steps.append(("clf", estimator))
        return Pipeline(steps), classifier_name, "none", selector_name
    if sampler == "smote-rus":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:
            raise ImportError("imbalanced-learn is not installed. Run: pip install imbalanced-learn") from exc
        steps = []
        if selector is not None:
            steps.append(("select", selector))
        steps.append(("scale", StandardScaler()))
        steps.extend(
            [
                ("smote", SMOTE(random_state=random_state, k_neighbors=3)),
                ("rus", RandomUnderSampler(random_state=random_state)),
            ]
        )
        steps.append(("clf", estimator))
        return ImbPipeline(steps), classifier_name, "smote-rus", selector_name
    raise ValueError("sampler must be one of: none, smote-rus")


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    classifier: str = "auto",
    sampler: str = "none",
    feature_selection: str = "ensemble",
    select_k: int = 67,
    random_state: int = 103,
) -> TrainingResult:
    """Train an epoch-independent baseline and evaluate a held-out split."""

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    if groups is None:
        groups = pd.Series(np.arange(len(y)), index=y.index)
    unique_groups = pd.Series(groups).nunique()
    class_counts = pd.Series(y_encoded).value_counts()
    if unique_groups >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y_encoded, groups))
        split_strategy = "subject_group_holdout"
    elif class_counts.min() >= 2:
        indices = np.arange(len(y_encoded))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.25,
            random_state=random_state,
            stratify=y_encoded,
        )
        split_strategy = "epoch_holdout_smoke_test"
    else:
        raise ValueError("Need at least two subjects for subject-wise validation, or at least two samples per class for a smoke split.")
    pipeline, classifier_name, sampler_name, selector_name = make_pipeline(
        classifier=classifier,
        sampler=sampler,
        feature_selection=feature_selection,
        select_k=select_k,
        random_state=random_state,
    )
    pipeline.fit(X.iloc[train_idx], y_encoded[train_idx])
    pred = pipeline.predict(X.iloc[test_idx])
    metrics = classification_metrics(y_encoded[test_idx], pred)
    selector = pipeline.named_steps.get("select")
    selected_feature_names = list(X.columns[selector.get_support()]) if selector is not None else list(X.columns)
    return TrainingResult(
        pipeline,
        label_encoder,
        list(X.columns),
        metrics,
        split_strategy,
        train_idx,
        test_idx,
        classifier_name,
        sampler_name,
        selector_name,
        selected_feature_names,
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    scores = np.asarray(scores, dtype=float)
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def save_training_result(result: TrainingResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, out)
    return out


def load_training_result(path: str | Path) -> TrainingResult:
    return joblib.load(path)
