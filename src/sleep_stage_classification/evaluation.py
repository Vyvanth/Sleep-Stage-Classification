"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .models import classification_metrics


def detailed_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    """Return per-class scores and confusion matrix as JSON-safe objects."""

    return {
        "classification_report": classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def comparison_tables(y_true: np.ndarray, predictions: dict[str, np.ndarray], class_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and per-class metric tables for several predictors."""

    overall_rows = []
    per_class_rows = []
    labels = list(range(len(class_names)))
    for model_name, y_pred in predictions.items():
        metrics = classification_metrics(y_true, y_pred)
        overall_rows.append({"model": model_name, **metrics})
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        for class_name in class_names:
            per_class_rows.append(
                {
                    "model": model_name,
                    "stage": class_name,
                    "precision": float(report[class_name]["precision"]),
                    "recall": float(report[class_name]["recall"]),
                    "f1": float(report[class_name]["f1-score"]),
                    "support": int(report[class_name]["support"]),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(per_class_rows)


def confusion_matrix_frame(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Return a labeled confusion matrix data frame."""

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(matrix, index=[f"true_{name}" for name in class_names], columns=[f"pred_{name}" for name in class_names])
