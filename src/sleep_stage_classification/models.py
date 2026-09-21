"""Baseline classifier training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


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


def make_feature_selector(feature_selection: str = "mutual_info", select_k: int = 40, random_state: int = 103):
    """Create the supervised feature-selection stage used during training."""

    if feature_selection == "none":
        return None, "none"
    if feature_selection == "mutual_info":
        if select_k <= 0:
            raise ValueError("select_k must be positive.")
        return (
            SelectKBest(
                score_func=partial(mutual_info_classif, random_state=random_state),
                k=select_k,
            ),
            "mutual_info",
        )
    raise ValueError("feature_selection must be one of: none, mutual_info")


def make_pipeline(
    classifier: str = "auto",
    sampler: str = "none",
    feature_selection: str = "mutual_info",
    select_k: int = 40,
    random_state: int = 103,
):
    """Create a training pipeline with optional imbalance handling."""

    estimator, classifier_name = make_baseline_classifier(classifier=classifier, random_state=random_state)
    selector, selector_name = make_feature_selector(feature_selection, select_k, random_state)
    if sampler == "none":
        steps = [("scale", StandardScaler())]
        if selector is not None:
            steps.append(("select", selector))
        steps.append(("clf", estimator))
        return Pipeline(steps), classifier_name, "none", selector_name
    if sampler == "smote-rus":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:
            raise ImportError("imbalanced-learn is not installed. Run: pip install imbalanced-learn") from exc
        steps = [
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=random_state, k_neighbors=3)),
            ("rus", RandomUnderSampler(random_state=random_state)),
        ]
        if selector is not None:
            steps.append(("select", selector))
        steps.append(("clf", estimator))
        return ImbPipeline(steps), classifier_name, "smote-rus", selector_name
    raise ValueError("sampler must be one of: none, smote-rus")


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    classifier: str = "auto",
    sampler: str = "none",
    feature_selection: str = "mutual_info",
    select_k: int = 40,
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
