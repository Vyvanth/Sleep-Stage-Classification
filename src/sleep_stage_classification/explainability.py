"""Feature attribution utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_linear_attributions(
    feature_frame: pd.DataFrame,
    probabilities: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Create deterministic per-epoch attribution rows without SHAP.

    The fallback ranks standardized feature magnitudes weighted by prediction
    confidence. When SHAP is installed, production experiments can replace this
    with TreeExplainer values while preserving the same output schema.
    """

    X = feature_frame[feature_names].astype(float)
    z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    confidence = np.max(probabilities, axis=1)
    rows = []
    for row_idx, (_, row) in enumerate(z.iterrows()):
        scores = row.abs().sort_values(ascending=False).head(top_k)
        for rank, (feature, magnitude) in enumerate(scores.items(), 1):
            rows.append(
                {
                    "epoch_index": int(feature_frame.iloc[row_idx].get("epoch_index", row_idx)),
                    "rank": rank,
                    "feature": feature,
                    "attribution": float(magnitude * confidence[row_idx]),
                    "direction": "high" if row[feature] >= 0 else "low",
                }
            )
    return pd.DataFrame(rows)
