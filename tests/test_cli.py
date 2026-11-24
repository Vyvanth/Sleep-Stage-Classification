import argparse
from sleep_stage_classification.cli import cmd_train


def test_train_loads_lightgbm_parameters_from_tuning_file(monkeypatch):
    captured = {}

    def fake_train_and_write(*args, **kwargs):
        captured["params"] = args[-1]

    monkeypatch.setattr("sleep_stage_classification.cli.train_and_write", fake_train_and_write)
    monkeypatch.setattr("pathlib.Path.read_text", lambda *args, **kwargs: '{"best_lightgbm_params": {"num_leaves": 15}}')
    args = argparse.Namespace(
        lightgbm_params=None,
        lightgbm_params_file="outputs/lightgbm_tuning_99.json",
        features="features.csv",
        model_out="model.joblib",
        metrics_out="metrics.json",
        transition_weight=0.35,
        classifier="lightgbm",
        sampler="smote-rus",
        feature_selection="ensemble",
        select_k=67,
        context_lags=1,
    )

    cmd_train(args)

    assert captured["params"] == {"num_leaves": 15}
