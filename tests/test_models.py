import numpy as np
import pandas as pd

from sleep_stage_classification.models import (
    ProbabilityCalibrator,
    StageEncoder,
    cross_fitted_calibrated_probabilities,
    modality_feature_columns,
    train_baseline,
    tune_lightgbm,
)


def test_stage_encoder_uses_fixed_project_stage_ids():
    encoder = StageEncoder()
    encoded = encoder.fit_transform(["W", "N1", "N2", "N3", "REM"])

    assert encoded.tolist() == [0, 1, 2, 3, 4]
    assert encoder.inverse_transform([0, 1, 2, 3, 4]).tolist() == ["W", "N1", "N2", "N3", "REM"]


def test_modality_feature_columns_preserve_channel_prefixed_features():
    X = pd.DataFrame(columns=["EEG_Fpz_delta_power", "EOG_horizontal_tkeo_mean", "EMG_submental_rms"])
    assert modality_feature_columns(X, "EEG") == ["EEG_Fpz_delta_power"]
    assert modality_feature_columns(X, "EOG") == ["EOG_horizontal_tkeo_mean"]
    assert modality_feature_columns(X, "EMG") == ["EMG_submental_rms"]


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
        feature_selection="ensemble",
        select_k=3,
    )

    assert result.feature_selection_name == "ensemble_lgbm_mi_permutation"
    assert len(result.selected_feature_names) == 3
    assert result.model.named_steps["select"].cv_strategy_ == "subject_group_kfold"


def test_lightgbm_tuning_uses_grouped_folds():
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(90, 6)), columns=[f"feature_{idx}" for idx in range(6)])
    y = pd.Series(np.tile(["W", "N1", "N2"], 30))
    groups = pd.Series(np.repeat(["subject_a", "subject_b", "subject_c"], 30))

    result = tune_lightgbm(
        X,
        y,
        groups,
        sampler="none",
        feature_selection="none",
        cv_folds=3,
        candidates=[{"n_estimators": 10, "learning_rate": 0.1, "num_leaves": 7}],
    )

    assert result.cv_folds == 3
    assert result.best_params["num_leaves"] == 7
    assert len(result.candidates[0]["folds"]) == 3


def test_probability_calibration_preserves_valid_probability_rows():
    probabilities = np.array(
        [[0.70, 0.20, 0.10], [0.65, 0.25, 0.10], [0.10, 0.70, 0.20],
         [0.10, 0.60, 0.30], [0.20, 0.10, 0.70], [0.15, 0.15, 0.70]],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    groups = np.array(["a", "a", "b", "b", "c", "c"])

    cross_fitted = cross_fitted_calibrated_probabilities(probabilities, labels, groups, cv_folds=3)
    fitted = ProbabilityCalibrator(n_states=3).fit(probabilities, labels).transform(probabilities)

    assert np.allclose(cross_fitted.sum(axis=1), 1.0)
    assert np.allclose(fitted.sum(axis=1), 1.0)
    assert cross_fitted.shape == probabilities.shape
