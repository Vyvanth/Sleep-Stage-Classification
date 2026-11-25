"""Command-line workflows for the sleep staging project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_TO_STAGE, STAGE_ORDER
from .data import build_sleep_edf_feature_csv, load_feature_csv, make_synthetic_feature_csv
from .features import add_sequential_context
from .evaluation import comparison_tables, confusion_matrix_frame
from .explainability import tree_shap_attributions
from .gemini_reporting import generate_gemini_report
from .models import (
    ProbabilityCalibrator,
    cross_fitted_calibrated_probabilities,
    load_training_result,
    modality_feature_columns,
    predict_proba,
    save_training_result,
    subject_holdout_split,
    train_baseline,
    tune_lightgbm,
    oof_probabilities,
)
from .report_evaluation import evaluate_report
from .reporting import generate_grounded_report
from .temporal import (
    default_sleep_transition_matrix,
    estimate_transition_by_record,
    smooth_by_record as smooth_probability_sequences,
    viterbi_smooth,
)
from .temporal_models import TorchLinearChainCRF, WindowBiLSTM


def cmd_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = make_synthetic_feature_csv(output_dir / "synthetic_features.csv")
    train_and_write(
        features_path,
        output_dir / "baseline.joblib",
        output_dir / "metrics.json",
        output_dir,
        args.transition_weight,
        args.classifier,
        args.sampler,
        args.feature_selection,
        args.select_k,
        context_lags=getattr(args, "context_lags", 1),
    )


def cmd_train(args: argparse.Namespace) -> None:
    if args.lightgbm_params and args.lightgbm_params_file:
        raise ValueError("Use either --lightgbm-params or --lightgbm-params-file, not both.")
    if args.lightgbm_params_file:
        parameter_file = Path(args.lightgbm_params_file)
        parameter_data = json.loads(parameter_file.read_text(encoding="utf-8"))
        lightgbm_params = parameter_data.get("best_lightgbm_params", parameter_data)
        if not isinstance(lightgbm_params, dict):
            raise ValueError("The parameter file must contain a JSON object or 'best_lightgbm_params' object.")
    else:
        lightgbm_params = json.loads(args.lightgbm_params) if args.lightgbm_params else None
    train_and_write(
        Path(args.features),
        Path(args.model_out),
        Path(args.metrics_out),
        Path(args.metrics_out).parent,
        args.transition_weight,
        args.classifier,
        args.sampler,
        args.feature_selection,
        args.select_k,
        lightgbm_params,
        context_lags=args.context_lags,
    )


def cmd_extract_sleep_edf(args: argparse.Namespace) -> None:
    output_path = build_sleep_edf_feature_csv(
        root=args.root,
        output_path=args.output,
        channels=args.channels,
        max_records=args.max_records,
        sample_rate_hz=args.sample_rate_hz,
        trim_wake_minutes=args.trim_wake_minutes,
        denoise=not args.no_denoise,
        epoch_output_dir=args.epoch_output_dir,
        denoising_plot_dir=args.denoising_plot_dir,
    )
    print(f"Wrote Sleep-EDF features: {output_path}")


def train_and_write(
    features_path: Path,
    model_out: Path,
    metrics_out: Path,
    output_dir: Path,
    transition_weight: float = 0.35,
    classifier: str = "auto",
    sampler: str = "none",
    feature_selection: str = "ensemble",
    select_k: int = 67,
    lightgbm_params: dict[str, float | int] | None = None,
    context_lags: int = 1,
) -> None:
    X, y, metadata = load_feature_csv(features_path)
    result = train_baseline(
        X,
        y,
        groups=metadata["subject_id"],
        classifier=classifier,
        sampler=sampler,
        feature_selection=feature_selection,
        select_k=select_k,
        lightgbm_params=lightgbm_params,
        context_lags=context_lags,
        metadata=metadata,
    )
    X_model = add_sequential_context(X, metadata, lags=context_lags) if context_lags > 0 else X
    probabilities = predict_proba(result.model, X_model)
    y_encoded = result.label_encoder.transform(y)
    transition = estimate_transition_by_record(
        y_encoded[result.train_indices],
        metadata["record_id"].to_numpy()[result.train_indices],
        metadata["epoch_index"].to_numpy()[result.train_indices],
        n_states=len(result.label_encoder.classes_),
    )
    temporal_pred_ids = smooth_by_record(probabilities, metadata, transition, transition_weight=transition_weight)
    baseline_pred_ids = np.argmax(probabilities, axis=1)
    baseline_stages = result.label_encoder.inverse_transform(baseline_pred_ids)
    temporal_stages = result.label_encoder.inverse_transform(temporal_pred_ids)
    predictions = metadata.copy()
    predictions["true_stage"] = y.values
    predictions["baseline_stage"] = baseline_stages
    predictions["predicted_stage"] = temporal_stages
    predictions["split"] = "unused"
    predictions.loc[result.train_indices, "split"] = "train"
    predictions.loc[result.test_indices, "split"] = "test"
    predictions_path = output_dir / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    report = generate_grounded_report(predictions)
    (output_dir / "sleep_report.txt").write_text(report, encoding="utf-8")
    test_true = y_encoded[result.test_indices]
    test_baseline = baseline_pred_ids[result.test_indices]
    test_temporal = temporal_pred_ids[result.test_indices]
    class_names = list(result.label_encoder.classes_)
    overall_table, per_class_table = comparison_tables(
        test_true,
        {"baseline_epoch_independent": test_baseline, "temporal_viterbi": test_temporal},
        class_names,
    )
    metrics = {
        "split_strategy": result.split_strategy,
        "classes": class_names,
        "classifier": result.classifier_name,
        "sampler": result.sampler_name,
        "feature_selection": result.feature_selection_name,
        "selected_feature_count": len(result.selected_feature_names),
        "selected_features": result.selected_feature_names,
        "context_lags": context_lags,
        "transition_weight": transition_weight,
        "overall": overall_table.to_dict(orient="records"),
        "per_class": per_class_table.to_dict(orient="records"),
        "n1_f1_delta": f1_delta(per_class_table, stage="N1"),
        "temporal_prior": transition.tolist(),
        "fallback_temporal_prior": default_sleep_transition_matrix().tolist(),
    }
    metrics["split_strategy"] = result.split_strategy
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    selected_features_path = metrics_out.with_name(f"{metrics_out.stem}_selected_features.json")
    selected_features_path.write_text(
        json.dumps(
            {
                "feature_selection": result.feature_selection_name,
                "selected_feature_count": len(result.selected_feature_names),
                "selected_features": result.selected_feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    overall_path = output_dir / "baseline_vs_temporal_overall.csv"
    per_class_path = output_dir / "baseline_vs_temporal_per_class.csv"
    baseline_cm_path = output_dir / "confusion_matrix_baseline.csv"
    temporal_cm_path = output_dir / "confusion_matrix_temporal.csv"
    overall_table.to_csv(overall_path, index=False)
    per_class_table.to_csv(per_class_path, index=False)
    confusion_matrix_frame(test_true, test_baseline, class_names).to_csv(baseline_cm_path)
    confusion_matrix_frame(test_true, test_temporal, class_names).to_csv(temporal_cm_path)
    save_training_result(result, model_out)
    print(f"Wrote model: {model_out}")
    print(f"Wrote metrics: {metrics_out}")
    print(f"Wrote selected feature list: {selected_features_path}")
    print(f"Wrote comparison table: {overall_path}")
    print(f"Wrote per-class table: {per_class_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote report: {output_dir / 'sleep_report.txt'}")


def cmd_tune_lightgbm(args: argparse.Namespace) -> None:
    """Tune only on grouped folds and write reusable LightGBM parameters."""

    X, y, metadata = load_feature_csv(args.features)
    result = tune_lightgbm(
        X,
        y,
        metadata["subject_id"],
        sampler=args.sampler,
        feature_selection=args.feature_selection,
        select_k=args.select_k,
        cv_folds=args.cv_folds,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "selection_metric": "mean_macro_f1",
                "cv_strategy": "subject_group_kfold",
                "cv_folds": result.cv_folds,
                "best_lightgbm_params": result.best_params,
                "candidates": result.candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote LightGBM tuning results: {out}")
    print(f"Best parameters: {json.dumps(result.best_params)}")


def _load_lightgbm_params(path: str | None) -> dict[str, float | int] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters = data.get("best_lightgbm_params", data)
    if not isinstance(parameters, dict):
        raise ValueError("The parameter file must contain a JSON object or 'best_lightgbm_params' object.")
    return parameters


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare requested baselines and modality ablations on one subject holdout."""

    X, y, metadata = load_feature_csv(args.features)
    parameters = _load_lightgbm_params(args.lightgbm_params_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_features = modality_feature_columns(X, "all")
    comparisons = [
        ("lightgbm_class_weights", "lightgbm", "none"),
        ("lightgbm_smote_rus", "lightgbm", "smote-rus"),
        ("extra_trees_class_weights", "extra_trees", "none"),
        ("extra_trees_smote_rus", "extra_trees", "smote-rus"),
    ]
    results: list[dict[str, object]] = []
    per_class_tables: list[pd.DataFrame] = []
    trained: dict[str, object] = {}
    for label, classifier, sampler in comparisons:
        result = train_baseline(
            X, y, groups=metadata["subject_id"], classifier=classifier, sampler=sampler,
            feature_selection=args.feature_selection, select_k=args.select_k,
            lightgbm_params=parameters if classifier == "lightgbm" else None,
            feature_columns=all_features,
            context_lags=args.context_lags,
            metadata=metadata,
        )
        test_y = result.label_encoder.transform(y.iloc[result.test_indices])
        X_model = add_sequential_context(X[all_features], metadata, lags=args.context_lags) if args.context_lags > 0 else X[all_features]
        predicted = result.model.predict(X_model.iloc[result.test_indices])
        overall, per_class = comparison_tables(test_y, {label: predicted}, list(result.label_encoder.classes_))
        row = overall.iloc[0].to_dict()
        row.update({"condition": label, "split_strategy": result.split_strategy, "selected_feature_count": len(result.selected_feature_names)})
        results.append(row)
        per_class_tables.append(per_class.assign(condition=label))
        confusion_matrix_frame(test_y, predicted, list(result.label_encoder.classes_)).to_csv(output_dir / f"{label}_confusion_matrix.csv")
        trained[label] = result

    ablation_rows: list[dict[str, object]] = []
    for modality in ("EEG", "EOG", "EMG", "all"):
        columns = modality_feature_columns(X, modality)
        result = train_baseline(
            X, y, groups=metadata["subject_id"], classifier="lightgbm", sampler="smote-rus",
            feature_selection=args.feature_selection, select_k=min(args.select_k, len(columns)),
            lightgbm_params=parameters, feature_columns=columns,
            context_lags=args.context_lags,
            metadata=metadata,
        )
        test_y = result.label_encoder.transform(y.iloc[result.test_indices])
        X_model = add_sequential_context(X[columns], metadata, lags=args.context_lags) if args.context_lags > 0 else X[columns]
        predicted = result.model.predict(X_model.iloc[result.test_indices])
        row = {
            "modality": modality,
            "input_feature_count": len(columns),
            "selected_feature_count": len(result.selected_feature_names),
            **result.metrics,
        }
        ablation_rows.append(row)

    results_frame = pd.DataFrame(results).sort_values(["macro_f1", "accuracy"], ascending=False)
    results_frame.to_csv(output_dir / "baseline_benchmark.csv", index=False)
    pd.concat(per_class_tables, ignore_index=True).to_csv(output_dir / "baseline_benchmark_per_class.csv", index=False)
    pd.DataFrame(ablation_rows).sort_values(["macro_f1", "accuracy"], ascending=False).to_csv(
        output_dir / "modality_ablation.csv", index=False
    )
    best_label = str(results_frame.iloc[0]["condition"])
    best_result = trained[best_label]
    save_training_result(best_result, output_dir / "best_all_modalities_model.joblib")
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "selection_metric": "macro_f1",
                "best_condition": best_label,
                "best_metrics": best_result.metrics,
                "feature_selection": best_result.feature_selection_name,
                "selected_features": best_result.selected_feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote benchmark results: {output_dir / 'baseline_benchmark.csv'}")
    print(f"Wrote modality ablation: {output_dir / 'modality_ablation.csv'}")
    print(f"Saved best all-modalities model: {output_dir / 'best_all_modalities_model.joblib'}")


def _fill_sequence_edges(predictions: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Use epoch-wise predictions at sequence edges lacking a complete LSTM window."""

    filled = np.asarray(predictions, dtype=int).copy()
    filled[filled < 0] = np.asarray(fallback, dtype=int)[filled < 0]
    return filled


def cmd_temporal_benchmark(args: argparse.Namespace) -> None:
    """Compare leakage-free HMM, CRF, and BiLSTM temporal models on one subject holdout."""

    print("[1/7] Loading epoch features...", flush=True)
    X, y, metadata = load_feature_csv(args.features)
    parameters = _load_lightgbm_params(args.lightgbm_params_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_train_idx, final_test_idx, _ = subject_holdout_split(y, metadata["subject_id"])
    if args.tune_on_training:
        print("[tuning] Selecting LightGBM settings using training subjects only...", flush=True)
        tuning = tune_lightgbm(
            X.iloc[final_train_idx].reset_index(drop=True),
            y.iloc[final_train_idx].reset_index(drop=True),
            metadata.iloc[final_train_idx]["subject_id"].reset_index(drop=True),
            sampler="smote-rus",
            feature_selection=args.feature_selection,
            select_k=args.select_k,
            cv_folds=args.oof_folds,
            metadata=metadata.iloc[final_train_idx].reset_index(drop=True),
            context_lags=args.context_lags,
        )
        parameters = tuning.best_params
        (output_dir / "nested_lightgbm_tuning.json").write_text(
            json.dumps({"best_lightgbm_params": parameters, "candidates": tuning.candidates}, indent=2),
            encoding="utf-8",
        )
    print("[2/7] Fitting the subject-wise LightGBM baseline...", flush=True)
    result = train_baseline(
        X,
        y,
        groups=metadata["subject_id"],
        classifier="lightgbm",
        sampler="smote-rus",
        feature_selection=args.feature_selection,
        select_k=args.select_k,
        lightgbm_params=parameters,
        context_lags=args.context_lags,
        metadata=metadata,
        train_indices=final_train_idx,
        test_indices=final_test_idx,
    )
    train_idx, test_idx = result.train_indices, result.test_indices
    X_model = add_sequential_context(X, metadata, lags=args.context_lags) if args.context_lags > 0 else X
    X_train, X_test = X_model.iloc[train_idx], X_model.iloc[test_idx]
    y_encoded = result.label_encoder.transform(y)
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    train_metadata, test_metadata = metadata.iloc[train_idx], metadata.iloc[test_idx]
    train_records = train_metadata["record_id"].to_numpy()
    test_records = test_metadata["record_id"].to_numpy()
    train_epochs = train_metadata["epoch_index"].to_numpy()
    test_epochs = test_metadata["epoch_index"].to_numpy()

    # The temporal learners receive OOF probabilities only, never in-sample baseline scores.
    print("[3/7] Creating subject-wise out-of-fold probabilities...", flush=True)
    oof = oof_probabilities(
        X_train.reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        train_metadata["subject_id"].reset_index(drop=True),
        classifier="lightgbm",
        sampler="smote-rus",
        feature_selection=args.feature_selection,
        select_k=args.select_k,
        lightgbm_params=parameters,
        cv_folds=args.oof_folds,
    )
    test_probabilities = predict_proba(result.model, X_test)
    baseline_predictions = np.argmax(test_probabilities, axis=1)
    print("[4/7] Cross-fitting probability calibration and tuning HMM decoding...", flush=True)
    calibration_oof = cross_fitted_calibrated_probabilities(
        oof, y_train, train_metadata["subject_id"].to_numpy(), cv_folds=args.oof_folds
    )
    probability_calibrator = ProbabilityCalibrator(n_states=len(STAGE_ORDER)).fit(oof, y_train)
    calibrated_test_probabilities = probability_calibrator.transform(test_probabilities)
    calibrated_baseline_predictions = np.argmax(calibrated_test_probabilities, axis=1)
    transition = estimate_transition_by_record(
        y_train, train_records, train_epochs, n_states=len(STAGE_ORDER)
    )
    candidate_weights = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.60, 0.75, 1.0, 1.25]
    hmm_tuning_scores = []
    for weight in candidate_weights:
        candidate_predictions = smooth_probability_sequences(
            calibration_oof, train_records, train_epochs, transition, weight
        )
        candidate_overall, _ = comparison_tables(
            y_train, {"hmm": candidate_predictions}, list(result.label_encoder.classes_)
        )
        hmm_tuning_scores.append({"transition_weight": weight, **candidate_overall.iloc[0].to_dict()})
    tuned_weight = max(hmm_tuning_scores, key=lambda item: item[args.hmm_selection_metric])["transition_weight"]
    hmm_predictions = smooth_probability_sequences(
        calibrated_test_probabilities, test_records, test_epochs, transition, tuned_weight
    )

    print("[5/7] Training the linear-chain CRF...", flush=True)
    crf = TorchLinearChainCRF(n_states=len(STAGE_ORDER)).fit(
        calibration_oof, y_train, train_records, train_epochs, epochs=args.crf_epochs
    )
    crf_predictions = crf.predict(calibrated_test_probabilities, test_records, test_epochs)

    print("[6/7] Training the window BiLSTM...", flush=True)
    selected = result.selected_feature_names
    bilstm = WindowBiLSTM(
        feature_count=len(selected),
        window_size=args.lstm_window,
        batch_size=args.lstm_batch_size,
    )
    bilstm.fit(
        X_train[selected], y_train, train_records, train_epochs,
        epochs=args.lstm_epochs, patience=args.lstm_patience,
    )
    bilstm_predictions = _fill_sequence_edges(
        bilstm.predict(X_test[selected], test_records, test_epochs), baseline_predictions
    )

    print("[7/7] Writing comparison tables and confusion matrices...", flush=True)
    class_names = list(result.label_encoder.classes_)
    predictions = {
        "baseline_epoch_independent": baseline_predictions,
        "calibrated_epoch_independent": calibrated_baseline_predictions,
        "hmm_viterbi": hmm_predictions,
        "linear_chain_crf": crf_predictions,
        "window_bilstm": bilstm_predictions,
    }
    overall, per_class = comparison_tables(y_test, predictions, class_names)
    overall.to_csv(output_dir / "temporal_comparison_overall.csv", index=False)
    per_class.to_csv(output_dir / "temporal_comparison_per_class.csv", index=False)
    for name, values in predictions.items():
        confusion_matrix_frame(y_test, values, class_names).to_csv(output_dir / f"{name}_confusion_matrix.csv")
    test_predictions = test_metadata.copy()
    test_predictions["true_stage"] = result.label_encoder.inverse_transform(y_test)
    for name, values in predictions.items():
        test_predictions[name] = result.label_encoder.inverse_transform(values)
    test_predictions.to_csv(output_dir / "temporal_test_predictions.csv", index=False)
    n1 = per_class[per_class["stage"] == "N1"][["model", "f1"]].to_dict(orient="records")
    summary = {
        "split_strategy": result.split_strategy,
        "baseline_classifier": result.classifier_name,
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "oof_folds": min(args.oof_folds, train_metadata["subject_id"].nunique()),
        "hmm_transition_weight": tuned_weight,
        "hmm_transition_matrix": transition.tolist(),
        "hmm_selection_metric": args.hmm_selection_metric,
        "hmm_training_oof_candidates": hmm_tuning_scores,
        "probability_calibration": "cross-fitted multinomial logistic calibration on baseline OOF probabilities",
        "lightgbm_params": parameters,
        "nested_tuning": args.tune_on_training,
        "bilstm_window_size": args.lstm_window,
        "bilstm_batch_size": args.lstm_batch_size,
        "bilstm_device": bilstm.device_name_,
        "bilstm_validation_macro_f1": bilstm.validation_macro_f1_,
        "n1_f1": n1,
        "overall": overall.to_dict(orient="records"),
    }
    (output_dir / "temporal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_training_result(result, output_dir / "baseline_model.joblib")
    print(f"Wrote temporal comparison: {output_dir / 'temporal_comparison_overall.csv'}")
    print(f"Wrote N1 F1 comparison: {output_dir / 'temporal_comparison_per_class.csv'}")
    print(f"Wrote temporal summary: {output_dir / 'temporal_summary.json'}")


def cmd_explain(args: argparse.Namespace) -> None:
    X, _, metadata = load_feature_csv(args.features)
    result = load_training_result(args.model)
    context_lags = getattr(result, "context_lags", 0)
    X_model = add_sequential_context(X, metadata, lags=context_lags) if context_lags > 0 else X
    probabilities = predict_proba(result.model, X_model)
    predicted_ids = np.argmax(probabilities, axis=1)
    predicted_stages = result.label_encoder.inverse_transform(predicted_ids)
    attributions = tree_shap_attributions(
        pd.concat([metadata.reset_index(drop=True), X_model.reset_index(drop=True)], axis=1),
        result.model,
        result.feature_names,
        result.selected_feature_names,
        predicted_ids,
        predicted_stages,
        top_k=args.top_k,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attributions.to_csv(output_path, index=False)
    print(f"Wrote SHAP attributions: {output_path}")


def smooth_by_record(probabilities: np.ndarray, metadata: pd.DataFrame, transition: np.ndarray, transition_weight: float = 0.35) -> np.ndarray:
    """Apply Viterbi smoothing independently to each recording."""

    smoothed = np.zeros(probabilities.shape[0], dtype=int)
    indexed = metadata.reset_index().sort_values(["record_id", "epoch_index"])
    for _, group in indexed.groupby("record_id", sort=False):
        indices = group["index"].to_numpy()
        smoothed[indices] = viterbi_smooth(probabilities[indices], transition, transition_weight=transition_weight)
    return smoothed


def f1_delta(per_class_table: pd.DataFrame, stage: str) -> float | None:
    """Return temporal minus baseline F1 for a target stage."""

    subset = per_class_table[per_class_table["stage"] == stage].set_index("model")
    required = {"baseline_epoch_independent", "temporal_viterbi"}
    if not required.issubset(subset.index):
        return None
    return float(subset.loc["temporal_viterbi", "f1"] - subset.loc["baseline_epoch_independent", "f1"])


def cmd_report(args: argparse.Namespace) -> None:
    predictions = pd.read_csv(args.predictions)
    attributions = pd.read_csv(args.attributions) if args.attributions else None
    predictions, attributions = select_record(predictions, attributions, args.record_id)
    report = generate_grounded_report(predictions, attributions)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote report: {out}")


def cmd_gemini_report(args: argparse.Namespace) -> None:
    predictions = pd.read_csv(args.predictions)
    attributions = pd.read_csv(args.attributions) if args.attributions else None
    predictions, attributions = select_record(predictions, attributions, args.record_id)
    report = generate_gemini_report(predictions, attributions, model_name=args.model)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote Gemini report: {out}")


def cmd_evaluate_report(args: argparse.Namespace) -> None:
    predictions = pd.read_csv(args.predictions)
    attributions = pd.read_csv(args.attributions)
    predictions, attributions = select_record(predictions, attributions, args.record_id)
    report = Path(args.report).read_text(encoding="utf-8")
    evaluation = evaluate_report(report, predictions, attributions, use_bertscore=args.use_bertscore)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Wrote report evaluation: {out}")


def select_record(
    predictions: pd.DataFrame,
    attributions: pd.DataFrame | None,
    record_id: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return one recording when requested so reports never mix subjects."""

    if not record_id:
        return predictions, attributions
    if "record_id" not in predictions:
        raise ValueError("The predictions file has no record_id column.")
    selected_predictions = predictions[predictions["record_id"] == record_id].copy()
    if selected_predictions.empty:
        raise ValueError(f"No predictions found for record_id={record_id!r}.")
    if attributions is not None and "record_id" in attributions:
        attributions = attributions[attributions["record_id"] == record_id].copy()
    return selected_predictions, attributions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sleep stage classification workflows")
    sub = parser.add_subparsers(required=True)
    demo = sub.add_parser("demo", help="Run an end-to-end synthetic smoke test")
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--transition-weight", type=float, default=0.35)
    demo.add_argument("--classifier", choices=["auto", "lightgbm", "extra_trees", "random_forest"], default="auto")
    demo.add_argument("--sampler", choices=["none", "smote-rus"], default="none")
    demo.add_argument("--feature-selection", choices=["none", "ensemble"], default="ensemble")
    demo.add_argument("--select-k", type=int, default=67)
    demo.add_argument("--context-lags", type=int, default=1, help="Use previous/next epoch spectral features from the same recording")
    demo.set_defaults(func=cmd_demo)
    extract = sub.add_parser("extract-sleep-edf", help="Extract epoch features from Sleep-EDF Expanded EDF files")
    extract.add_argument(
        "--root",
        default="data/raw/sleep-edf/sleep-edf-database-expanded-1.0.0",
        help="Sleep-EDF Expanded extracted dataset root",
    )
    extract.add_argument("--output", default="data/processed/sleep_edf_features.csv")
    extract.add_argument("--max-records", type=int, help="Limit records for the first smoke run")
    extract.add_argument("--sample-rate-hz", type=float, default=100.0)
    extract.add_argument("--trim-wake-minutes", type=int, default=30)
    extract.add_argument("--no-denoise", action="store_true", help="Skip wavelet denoising")
    extract.add_argument("--epoch-output-dir", help="Optional directory for compressed per-record epoch and label archives")
    extract.add_argument("--denoising-plot-dir", help="Optional directory for before/after wavelet denoising plots")
    extract.add_argument(
        "--channels",
        nargs="+",
        default=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"],
        help="Channel names to use when available",
    )
    extract.set_defaults(func=cmd_extract_sleep_edf)
    train = sub.add_parser("train", help="Train from an epoch feature CSV")
    train.add_argument("--features", required=True)
    train.add_argument("--model-out", required=True)
    train.add_argument("--metrics-out", required=True)
    train.add_argument("--transition-weight", type=float, default=0.35, help="Strength of temporal transition prior during Viterbi smoothing")
    train.add_argument("--classifier", choices=["auto", "lightgbm", "extra_trees", "random_forest"], default="auto")
    train.add_argument("--sampler", choices=["none", "smote-rus"], default="none")
    train.add_argument("--feature-selection", choices=["none", "ensemble"], default="ensemble")
    train.add_argument("--select-k", type=int, default=67, help="Maximum features retained after ensemble selection")
    train.add_argument("--lightgbm-params", help="JSON object of tuned LightGBM parameters, for example '{\"num_leaves\": 15}'")
    train.add_argument("--lightgbm-params-file", help="Path to tuning JSON written by tune-lightgbm")
    train.add_argument("--context-lags", type=int, default=1, help="Use previous/next epoch spectral features from the same recording")
    train.set_defaults(func=cmd_train)
    tune = sub.add_parser("tune-lightgbm", help="Tune LightGBM with subject-wise cross-validation")
    tune.add_argument("--features", required=True)
    tune.add_argument("--output", default="outputs/lightgbm_tuning.json")
    tune.add_argument("--cv-folds", type=int, default=3, help="Number of whole-subject validation folds")
    tune.add_argument("--sampler", choices=["none", "smote-rus"], default="smote-rus")
    tune.add_argument("--feature-selection", choices=["none", "ensemble"], default="ensemble")
    tune.add_argument("--select-k", type=int, default=67)
    tune.set_defaults(func=cmd_tune_lightgbm)
    benchmark = sub.add_parser("benchmark", help="Compare baseline models and EEG/EOG/EMG feature ablations")
    benchmark.add_argument("--features", required=True)
    benchmark.add_argument("--output-dir", default="outputs/benchmark")
    benchmark.add_argument("--lightgbm-params-file", help="Optional tuning JSON written by tune-lightgbm")
    benchmark.add_argument("--feature-selection", choices=["none", "ensemble"], default="ensemble")
    benchmark.add_argument("--select-k", type=int, default=67)
    benchmark.add_argument("--context-lags", type=int, default=1, help="Use previous/next epoch spectral features from the same recording")
    benchmark.set_defaults(func=cmd_benchmark)
    temporal_benchmark = sub.add_parser(
        "temporal-benchmark",
        help="Compare leakage-free HMM, CRF, and BiLSTM temporal models",
    )
    temporal_benchmark.add_argument("--features", required=True)
    temporal_benchmark.add_argument("--output-dir", default="outputs/temporal_benchmark")
    temporal_benchmark.add_argument("--lightgbm-params-file", help="Optional tuning JSON written by tune-lightgbm")
    temporal_benchmark.add_argument("--feature-selection", choices=["none", "ensemble"], default="ensemble")
    temporal_benchmark.add_argument("--select-k", type=int, default=67)
    temporal_benchmark.add_argument("--oof-folds", type=int, default=3, help="Whole-subject folds for baseline OOF probabilities")
    temporal_benchmark.add_argument(
        "--hmm-selection-metric",
        choices=["accuracy", "macro_f1", "cohen_kappa"],
        default="macro_f1",
        help="Training-only OOF metric used to select the HMM transition strength",
    )
    temporal_benchmark.add_argument("--crf-epochs", type=int, default=20)
    temporal_benchmark.add_argument("--lstm-window", type=int, default=11, help="Odd consecutive-epoch window size")
    temporal_benchmark.add_argument("--lstm-epochs", type=int, default=15)
    temporal_benchmark.add_argument("--lstm-patience", type=int, default=3)
    temporal_benchmark.add_argument("--lstm-batch-size", type=int, default=256, help="Mini-batch size for memory-safe BiLSTM training")
    temporal_benchmark.add_argument("--context-lags", type=int, default=1, help="Use previous/next epoch spectral features from the same recording")
    temporal_benchmark.add_argument("--tune-on-training", action="store_true", help="Tune LightGBM only within the final training subjects before evaluation")
    temporal_benchmark.set_defaults(func=cmd_temporal_benchmark)
    explain = sub.add_parser("explain", help="Generate TreeSHAP explanations for a trained baseline model")
    explain.add_argument("--features", required=True)
    explain.add_argument("--model", required=True)
    explain.add_argument("--output", required=True)
    explain.add_argument("--top-k", type=int, default=5, help="Explanatory features retained per epoch")
    explain.set_defaults(func=cmd_explain)
    report = sub.add_parser("report", help="Generate a grounded report from predictions")
    report.add_argument("--predictions", required=True)
    report.add_argument("--attributions")
    report.add_argument("--report-out", required=True)
    report.add_argument("--record-id", help="Generate a report for one Sleep-EDF recording")
    report.set_defaults(func=cmd_report)
    gemini_report = sub.add_parser("gemini-report", help="Generate a Gemini report from one recording's metrics and SHAP evidence")
    gemini_report.add_argument("--predictions", required=True)
    gemini_report.add_argument("--attributions", required=True)
    gemini_report.add_argument("--record-id", required=True, help="Sleep-EDF recording to summarize")
    gemini_report.add_argument("--report-out", required=True)
    gemini_report.add_argument("--model", default="gemini-3.6-flash")
    gemini_report.set_defaults(func=cmd_gemini_report)
    evaluate = sub.add_parser("evaluate-report", help="Evaluate factual coverage and optional BERTScore for one Gemini report")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--attributions", required=True)
    evaluate.add_argument("--report", required=True)
    evaluate.add_argument("--record-id", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--use-bertscore", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
