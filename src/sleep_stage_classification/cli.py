"""Command-line workflows for the sleep staging project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_TO_STAGE, STAGE_ORDER
from .data import build_sleep_edf_feature_csv, load_feature_csv, make_synthetic_feature_csv
from .evaluation import comparison_tables, confusion_matrix_frame
from .explainability import top_linear_attributions
from .models import predict_proba, save_training_result, train_baseline
from .reporting import generate_grounded_report
from .temporal import default_sleep_transition_matrix, estimate_transition_matrix, viterbi_smooth


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
    )


def cmd_train(args: argparse.Namespace) -> None:
    train_and_write(
        Path(args.features),
        Path(args.model_out),
        Path(args.metrics_out),
        Path(args.metrics_out).parent,
        args.transition_weight,
        args.classifier,
        args.sampler,
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
) -> None:
    X, y, metadata = load_feature_csv(features_path)
    result = train_baseline(X, y, groups=metadata["subject_id"], classifier=classifier, sampler=sampler)
    probabilities = predict_proba(result.model, X)
    y_encoded = result.label_encoder.transform(y)
    transition = estimate_transition_matrix(y_encoded[result.train_indices], len(result.label_encoder.classes_))
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
    attributions = top_linear_attributions(
        pd.concat([metadata, X], axis=1),
        probabilities,
        result.feature_names,
    )
    attributions_path = output_dir / "attributions.csv"
    attributions.to_csv(attributions_path, index=False)
    report = generate_grounded_report(predictions, attributions)
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
    print(f"Wrote comparison table: {overall_path}")
    print(f"Wrote per-class table: {per_class_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote report: {output_dir / 'sleep_report.txt'}")


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
    report = generate_grounded_report(predictions, attributions)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote report: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sleep stage classification workflows")
    sub = parser.add_subparsers(required=True)
    demo = sub.add_parser("demo", help="Run an end-to-end synthetic smoke test")
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--transition-weight", type=float, default=0.35)
    demo.add_argument("--classifier", choices=["auto", "lightgbm", "random_forest"], default="auto")
    demo.add_argument("--sampler", choices=["none", "smote-rus"], default="none")
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
    train.add_argument("--classifier", choices=["auto", "lightgbm", "random_forest"], default="auto")
    train.add_argument("--sampler", choices=["none", "smote-rus"], default="none")
    train.set_defaults(func=cmd_train)
    report = sub.add_parser("report", help="Generate a grounded report from predictions")
    report.add_argument("--predictions", required=True)
    report.add_argument("--attributions")
    report.add_argument("--report-out", required=True)
    report.set_defaults(func=cmd_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
