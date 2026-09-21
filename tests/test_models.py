import numpy as np
import pandas as pd

from sleep_stage_classification.models import train_baseline


def test_training_selects_requested_number_of_features():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(60, 6)), columns=[f"feature_{idx}" for idx in range(6)])
    y = pd.Series(np.tile(["W", "N1", "N2"], 20))
    groups = pd.Series(np.repeat(["subject_a", "subject_b", "subject_c"], 20))

    result = train_baseline(
        X,
        y,
        groups=groups,
        classifier="random_forest",
        feature_selection="mutual_info",
        select_k=3,
    )

    assert result.feature_selection_name == "mutual_info"
    assert len(result.selected_feature_names) == 3
