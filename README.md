# Sleep Stage Classification

Temporally-aware, explainable multimodal sleep stage classification with
LLM-grounded reporting and cross-dataset validation.

This repository implements the project plan from Review 1:

- preprocess PSG signals from EEG, EOG, and EMG channels
- extract time, frequency, Hjorth, entropy, and transient-energy features
- normalize each subject recording and use five-fold ensemble feature selection
- train an epoch-independent baseline classifier
- apply temporal post-processing over consecutive epochs
- generate per-epoch feature attributions
- produce sleep reports grounded in model evidence
- compare Sleep-EDF performance with ISRUC-Sleep generalization

The real Sleep-EDF and ISRUC datasets are not committed to the repository.
Place downloaded EDF files under `data/raw/` and use the CSV feature workflow
or extend the EDF loader mapping in `src/sleep_stage_classification/data.py`.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m sleep_stage_classification.cli demo --output-dir outputs/demo
pytest
```

The demo command creates a synthetic PSG-like dataset, trains a baseline model,
applies temporal smoothing, writes metrics, and generates an evidence-grounded
text report. It is intended as a smoke test until the public datasets are
downloaded.

## Train On Feature CSV

To extract features directly from the downloaded Sleep-EDF Expanded folder:

```powershell
python -m sleep_stage_classification.cli extract-sleep-edf `
  --max-records 5 `
  --output data/processed/sleep_edf_features_5_records.csv
```

Increase `--max-records` gradually: 5, then 20, then 50, then the full dataset.

The current feature protocol supersedes the earlier 99-feature experiment. It
uses Welch PSD for delta, theta, alpha, sigma, beta, and gamma absolute/relative
power, spectral entropy and band ratios, complete time-domain and Hjorth
features, EEG-only permutation entropy, and EOG/EMG TKEO statistics. This gives
116 raw features for the standard four-channel record before selecting 67.
Re-extract the full dataset after updating the feature code so the new columns
are available:

```powershell
python -m sleep_stage_classification.cli extract-sleep-edf `
  --output data/processed/sleep_edf_features_full_116.csv
```

Prepare a CSV where each row is one 30-second epoch. Required columns:

- `subject_id`
- `record_id`
- `epoch_index`
- `stage`
- feature columns such as `EEG_Fpz-Cz_delta_power`

Then run:

```powershell
python -m sleep_stage_classification.cli train `
  --features data/processed/sleep_edf_features.csv `
  --model-out models/sleep_edf_baseline.joblib `
  --metrics-out outputs/sleep_edf_metrics.json
```

Training writes:

- `metrics.json`: baseline vs temporal summary, per-class metrics, N1 F1 delta
- `baseline_vs_temporal_overall.csv`: accuracy, macro F1, and Cohen's Kappa
- `baseline_vs_temporal_per_class.csv`: precision, recall, F1, and support by stage
- `confusion_matrix_baseline.csv`
- `confusion_matrix_temporal.csv`
- `predictions.csv`

To match the Review 1 plan more closely, install the optional classifier stack
and train with LightGBM plus hybrid sampling:

```powershell
pip install lightgbm imbalanced-learn
python -m sleep_stage_classification.cli train `
  --features data/processed/sleep_edf_features_50_records.csv `
  --model-out models/lightgbm_smote_rus_50_records.joblib `
  --metrics-out outputs/metrics_lightgbm_smote_rus_50_records.json `
  --classifier lightgbm `
  --sampler smote-rus `
  --feature-selection ensemble `
  --select-k 67
```

The default `ensemble` selector calculates LightGBM gain, mutual information,
and macro-F1 permutation importance in five stratified folds on the training
partition only. It rewards features that rank consistently across folds,
applies the configured EEG/EOG/EMG weights (0.5/0.3/0.2), and removes features
correlated above 0.9 before retaining up to 67 features.

## Tune LightGBM

Tune before the final model run. The command keeps every subject entirely in a
training or validation fold and selects the candidate with the best mean macro
F1. It can take considerable time on the complete dataset because feature
selection is repeated inside each fold.

```powershell
python -m sleep_stage_classification.cli tune-lightgbm `
  --features data/processed/sleep_edf_features_full_116.csv `
  --output outputs/lightgbm_tuning_116.json `
  --cv-folds 3 `
  --sampler smote-rus `
  --feature-selection ensemble `
  --select-k 67
```

Use the tuning file directly in the final training command. This avoids
PowerShell quoting problems with JSON parameters:

```powershell
python -m sleep_stage_classification.cli train `
  --features data/processed/sleep_edf_features_full_116.csv `
  --model-out models/lightgbm_smote_rus_full_116.joblib `
  --metrics-out outputs/metrics_lightgbm_smote_rus_full_116.json `
  --classifier lightgbm `
  --sampler smote-rus `
  --feature-selection ensemble `
  --select-k 67 `
  --lightgbm-params-file outputs/lightgbm_tuning_116.json
```

## Baseline Benchmark And Modality Ablation

Run the four requested baseline conditions on one fixed, stratified
subject-level holdout: LightGBM with class weights, LightGBM with SMOTE-RUS,
ExtraTrees with class weights, and ExtraTrees with SMOTE-RUS. The same command
also writes EEG-only, EOG-only, EMG-only, and all-modalities ablation results.

```powershell
python -m sleep_stage_classification.cli benchmark `
  --features data/processed/sleep_edf_features_full_116.csv `
  --output-dir outputs/benchmark_116 `
  --lightgbm-params-file outputs/lightgbm_tuning_116.json `
  --feature-selection ensemble `
  --select-k 67
```

This produces `baseline_benchmark.csv`, `baseline_benchmark_per_class.csv`,
one confusion-matrix CSV per condition, `modality_ablation.csv`, and
`best_all_modalities_model.joblib`. The saved model was fitted only on the
training subjects; the held-out subjects remain reserved for final reporting.

## Temporal Model Benchmark

The temporal benchmark keeps one final set of subjects unseen. It first makes
subject-wise out-of-fold baseline probabilities for the training subjects,
then trains the temporal models from those probabilities. It compares the
epoch-independent baseline with a tuned HMM/Viterbi decoder, a linear-chain
CRF, and a class-weighted window BiLSTM. Sequence boundaries are never joined
across recordings, and the headline N1 F1 values are written per model.

```powershell
python -m sleep_stage_classification.cli temporal-benchmark `
  --features data/processed/sleep_edf_features_full_116.csv `
  --output-dir outputs/temporal_benchmark_116 `
  --lightgbm-params-file outputs/lightgbm_tuning_116.json `
  --feature-selection ensemble `
  --select-k 67 `
  --oof-folds 3 `
  --lstm-window 11 `
  --lstm-epochs 15
```

Read `temporal_comparison_overall.csv` for accuracy, macro F1, and Cohen's
Kappa. Read `temporal_comparison_per_class.csv` and filter `stage = N1` for
the main temporal-model result. `temporal_summary.json` preserves the selected
features, tuned HMM transition weight, and BiLSTM validation macro F1.

Generate per-epoch TreeSHAP explanations for a trained model:

```powershell
python -m sleep_stage_classification.cli explain `
  --features data/processed/sleep_edf_features_full.csv `
  --model models/lightgbm_smote_rus_full.joblib `
  --output outputs/attributions.csv `
  --top-k 5
```

To generate a Gemini report, set `GEMINI_API_KEY` in your Windows user environment variables, open a new terminal, and run:

```powershell
python -m sleep_stage_classification.cli gemini-report `
  --predictions outputs/predictions.csv `
  --attributions outputs/attributions.csv `
  --record-id SC4001 `
  --report-out outputs/SC4001_gemini_report.txt
```

Evaluate the report's factual coverage against its model metrics and SHAP evidence:

```powershell
python -m sleep_stage_classification.cli evaluate-report `
  --predictions outputs/predictions.csv `
  --attributions outputs/attributions.csv `
  --report outputs/SC4001_gemini_report.txt `
  --record-id SC4001 `
  --output outputs/SC4001_report_evaluation.json
```

Add `--use-bertscore` after installing the optional `bert-score` dependency to calculate semantic similarity against the deterministic evidence-grounded reference report.

## Generate A Report

```powershell
python -m sleep_stage_classification.cli report `
  --predictions outputs/demo/predictions.csv `
  --attributions outputs/demo/attributions.csv `
  --report-out outputs/demo/sleep_report.txt
```

## Project Layout

```text
src/sleep_stage_classification/
  config.py          Project constants and stage labels
  data.py            CSV and EDF-oriented dataset helpers
  preprocessing.py   Wavelet/filtering-ready signal preprocessing
  features.py        Epoch feature engineering
  models.py          Baseline classifier training/evaluation
  temporal.py        Viterbi/HMM-style temporal smoothing
  temporal_models.py Linear-chain CRF and window BiLSTM temporal learners
  explainability.py  SHAP/permutation-style attribution helpers
  reporting.py       Evidence-grounded sleep report generation
  evaluation.py      Metrics and comparison utilities
  cli.py             Command-line workflows
tests/               Focused regression tests
```

## Notes

Optional packages such as `lightgbm`, `imbalanced-learn`, `PyWavelets`, `MNE`,
`hmmlearn`, `shap`, and LLM SDKs are used when installed. The core pipeline has
fallbacks so development can continue in a basic Python environment.
