"""SHAP feature-attribution utilities for trained tree classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def tree_shap_attributions(
    feature_frame: pd.DataFrame,
    model,
    feature_names: list[str],
    selected_feature_names: list[str],
    predicted_class_ids: np.ndarray,
    predicted_stages: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return top TreeSHAP explanations for each baseline model prediction."""

    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Run: pip install shap") from exc

    X = feature_frame[feature_names].astype(float)
    selector = model.named_steps.get("select")
    if selector is not None:
        X = selector.transform(X)
    transformed = model.named_steps["scale"].transform(X)

    if transformed.shape[1] != len(selected_feature_names):
        raise ValueError("Selected feature names do not match the trained pipeline.")

    values = _as_sample_feature_class_values(
        shap.TreeExplainer(model.named_steps["clf"]).shap_values(transformed),
        sample_count=len(feature_frame),
        feature_count=len(selected_feature_names),
    )

    rows: list[dict[str, object]] = []
    metadata_columns = [column for column in ("subject_id", "record_id", "epoch_index") if column in feature_frame]
    for row_idx, class_id in enumerate(predicted_class_ids):
        class_values = values[row_idx, :, int(class_id)]
        top_indices = np.argsort(np.abs(class_values))[::-1][:top_k]
        metadata = {column: feature_frame.iloc[row_idx][column] for column in metadata_columns}
        for rank, feature_idx in enumerate(top_indices, 1):
            attribution = float(class_values[feature_idx])
            rows.append(
                {
                    **metadata,
                    "predicted_stage": str(predicted_stages[row_idx]),
                    "rank": rank,
                    "feature": selected_feature_names[feature_idx],
                    "attribution": attribution,
                    "direction": "increases" if attribution >= 0 else "decreases",
                }
            )
    return pd.DataFrame(rows)


def _as_sample_feature_class_values(values, sample_count: int, feature_count: int) -> np.ndarray:
    """Normalize SHAP's version-dependent multiclass output to N x F x C."""

    if isinstance(values, list):
        array = np.stack(values, axis=-1)
    else:
        array = np.asarray(values)

    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError(f"Unexpected SHAP value shape: {array.shape}")
    if array.shape[0] != sample_count:
        raise ValueError(f"SHAP output has {array.shape[0]} samples; expected {sample_count}.")
    if array.shape[1] == feature_count:
        return array
    if array.shape[2] == feature_count:
        return np.transpose(array, (0, 2, 1))
    raise ValueError(f"SHAP output has no feature axis of length {feature_count}: {array.shape}")
