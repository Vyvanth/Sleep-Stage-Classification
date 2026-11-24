"""Fill verified report placeholders and correct project-specific claims."""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX


SOURCE = Path(r"C:\Users\somis\Downloads\Project_Report_Phase1_Sleep_Stage_Classification (1).docx")
OUTPUT = Path("outputs/Project_Report_Phase1_Sleep_Stage_Classification_completed.docx")


def replace_paragraph(paragraph, text: str) -> None:
    """Replace visible paragraph text while retaining its paragraph formatting."""

    if paragraph.runs:
        first = paragraph.runs[0]
        first.text = text
        first.font.highlight_color = None
        for run in paragraph.runs[1:]:
            run.text = ""
            run.font.highlight_color = None
    else:
        paragraph.add_run(text)


def replace_cell(cell, text: str) -> None:
    replace_paragraph(cell.paragraphs[0], text)
    for paragraph in cell.paragraphs[1:]:
        replace_paragraph(paragraph, "")


def clear_highlights(document: Document) -> None:
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            run.font.highlight_color = None
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.highlight_color = None
    for section in document.sections:
        for paragraph in (*section.header.paragraphs, *section.footer.paragraphs):
            for run in paragraph.runs:
                run.font.highlight_color = None


def replace_if_contains(document: Document, marker: str, replacement: str) -> None:
    for paragraph in document.paragraphs:
        if marker in paragraph.text:
            replace_paragraph(paragraph, replacement)
            return
    raise ValueError(f"Marker not found: {marker}")


def main() -> None:
    document = Document(SOURCE)

    replacements = {
        "[FILL: number of levels]": (
            "where lambda is the threshold. Fine-scale coefficients dominated by noise are suppressed, while larger "
            "coefficients that carry the sleep-relevant waveform are kept. The wavelet family used is Daubechies 4 "
            "(db4), and the number of decomposition levels is seven."
        ),
        "[FILL: how much wake is retained": (
            "Sleep laboratory recordings often contain long periods of wakefulness before lights-out and after the "
            "final awakening. These periods are not informative and, if kept, make Wake by far the dominant class. "
            "Excess wake periods at the beginning and end of each recording are therefore trimmed so that at most "
            "30 minutes of wakefulness is retained at each end. The continuous signals are then cut into "
            "non-overlapping 30-second epochs, the standard unit of the AASM scoring manual, and each epoch is "
            "given the label of the annotation that covers it."
        ),
        "[FILL: summary statistics computed from the TKEO]": (
            "It tracks the instantaneous energy of a signal and responds strongly to short bursts, which makes it "
            "suited to detecting the rapid eye movements of REM sleep in the EOG and the brief muscle activity in "
            "the EMG. For each EOG and EMG epoch, the mean, maximum and variance of the absolute TKEO energy are computed."
        ),
        "[FILL: breakdown of how the 116 features": (
            "The 116 features comprise 32 time-domain features (8 features across 4 channels), 64 frequency-domain "
            "features (16 across 4 channels), 12 Hjorth features (3 across 4 channels), 2 EEG permutation-entropy "
            "features, and 6 TKEO features from EOG and EMG."
        ),
        "[FILL: weights used for EEG": (
            "3. Modality weighting. The combined score is multiplied by a weight that reflects domain knowledge: "
            "EEG, EOG and EMG features are assigned weights of 0.5, 0.3 and 0.2 respectively. EEG is favoured "
            "because it is the primary basis of conventional sleep-stage scoring rules."
        ),
        "[FILL: correlation threshold]": (
            "4. Correlation filtering. When two retained features have an absolute correlation above 0.90, the "
            "lower-ranked one is removed. This leaves a subset that is predictive and minimally redundant."
        ),
        "[CONFIRM: that feature scoring": (
            "Feature scoring, stability estimation, correlation filtering and selection are fitted within the "
            "training subjects only; the held-out subjects are not used to choose features. The procedure retains "
            "up to 67 features. In the reported full-data experiment, 64 non-redundant features remained after correlation filtering."
        ),
        "[FILL: tuned hyperparameters": (
            "Two ensemble classifiers are used as baselines. LightGBM is a gradient boosting framework that builds "
            "decision trees sequentially, with each new tree correcting the errors of the previous ones; it is fast, "
            "handles tabular features well and provides feature importance. ExtraTrees builds many randomised decision "
            "trees independently and averages their votes, which reduces variance. Both are well suited to a moderately "
            "sized, tabular feature set. LightGBM hyperparameters were selected by 3-fold subject-group cross-validation "
            "over four candidate configurations, using mean macro F1. The selected configuration used 500 trees, a "
            "learning rate of 0.025, 63 leaves, a minimum of 30 samples per leaf and L2 regularisation lambda of 1.0. "
            "ExtraTrees used 350 trees with balanced class weights."
        ),
        "[FILL: split scheme": (
            "The evaluation protocol is designed to prevent information from the evaluation subjects reaching the model "
            "during training. All Sleep-EDF subjects are divided into training subjects and held-out subjects, and every "
            "epoch of a subject goes to exactly one of the two groups. Models are fitted and class balancing is applied "
            "using training subjects only, and held-out subjects are used only to compute metrics. This matters because "
            "consecutive epochs from one person are highly similar: if epochs were split at random, near-identical epochs "
            "from the same night would fall on both sides of the split and the measured accuracy would be inflated. Since "
            "some subjects contributed more than one night, the split is made by subject identifier and not by recording. "
            "A stratified 4-fold group split was used to form the final holdout: 74 subjects (146 recordings) for training "
            "and 26 subjects (51 recordings) for evaluation."
        ),
        "[CONFIRM: how emission scores": (
            "where b_j(o_t) is the emission score of stage j for the observation at epoch t, and the best path is "
            "recovered by back-tracking. The emission scores are the five class probabilities output by the LightGBM "
            "baseline. The transition matrix is estimated from chronologically adjacent labels in the training hypnograms, "
            "without creating transitions across recordings. Its influence is tuned using training-subject out-of-fold "
            "probabilities; the selected transition weight was 0.60."
        ),
        "[CONFIRM: features supplied to the CRF": (
            "where f_k are feature functions on label transitions and observations, lambda_k their learned weights, and "
            "Z(x) a normalising constant. Unlike the HMM, the CRF learns the weights of transition and observation "
            "features jointly. The input observation at each epoch is the 5-dimensional, out-of-fold baseline probability "
            "vector. A linear emission layer and transition, start and end scores are trained with Adam (learning rate 0.01) "
            "for 20 epochs on complete recording sequences."
        ),
        "[FILL: window length, hidden size": (
            "A long short-term memory network processes a sequence while maintaining a memory cell, and a bidirectional "
            "LSTM reads the sequence both forwards and backwards, so each position is informed by both past and future "
            "context. In this module, an 11-epoch window of selected feature vectors is fed to a one-layer BiLSTM with "
            "32 hidden units in each direction, and the network predicts the central epoch. Training uses inverse-frequency "
            "weighted cross-entropy loss, Adam with a learning rate of 0.001, up to 15 epochs, and early stopping with "
            "a patience of three validation epochs."
        ),
        "[FILL: number of training and evaluation": (
            "All results are computed with the subject-wise protocol described in Section 5.6.3. The reported baseline "
            "is LightGBM with SMOTE and random undersampling, tuned by 3-fold subject-group cross-validation and evaluated "
            "on 26 held-out subjects (51 recordings; 61,166 epochs) after training on 74 subjects (146 recordings). "
            "The ensemble selector retained 64 non-redundant features from the 116 candidates. The temporal models operate "
            "on this baseline's output. All experiments use random seed 103."
        ),
        "[CONFIRM: wording once the exact protocol": (
            "The results in Table 7.5 are descriptive rather than a direct performance ranking because the feature "
            "configuration and validation protocol are not identical. This project uses a strict subject-wise holdout, "
            "so no epoch from an evaluation subject is available during training. A fair numerical comparison would "
            "require both approaches to be rerun on the same data using an identical subject-level split."
        ),
        "[FILL: two or three sentences interpreting the ablation": (
            "The all-modality model achieved the strongest performance, with an accuracy of 0.758 and macro F1 of 0.696. "
            "EEG was the strongest individual modality (macro F1 0.638), followed by EOG (0.603), while EMG alone was "
            "substantially weaker (0.379). The improvement from combining the modalities indicates that EOG and EMG add "
            "complementary information to the EEG features."
        ),
    }
    for marker, replacement in replacements.items():
        replace_if_contains(document, marker, replacement)

    # Correct claims that should say "up to 67" rather than promise exactly 67 features.
    for paragraph in document.paragraphs:
        if "selects 67 of them" in paragraph.text:
            replace_paragraph(paragraph, paragraph.text.replace("selects 67 of them", "selects up to 67 of them (64 in the reported full-data run)"))
        elif "a vector of 67 numbers" in paragraph.text:
            replace_paragraph(paragraph, paragraph.text.replace("a vector of 67 numbers", "a vector of up to 67 selected numbers (64 in the reported run)"))
        elif "a compact subset of 67 features" in paragraph.text:
            replace_paragraph(paragraph, paragraph.text.replace("a compact subset of 67 features", "a compact subset of up to 67 features (64 in the reported run)"))
        elif "reduces the set to 67 features" in paragraph.text:
            replace_paragraph(paragraph, paragraph.text.replace("reduces the set to 67 features", "reduces the set to at most 67 features"))
        elif "reduces it to 67 features" in paragraph.text:
            replace_paragraph(paragraph, paragraph.text.replace("reduces it to 67 features", "reduces it to a non-redundant subset of up to 67 features (64 in the reported run)"))
        elif paragraph.text.startswith("Figure 5.1 shows a quality check"):
            replace_paragraph(
                paragraph,
                "Figure 5.1 shows a quality check of wavelet denoising on one 30-second Sleep-EDF epoch. In each "
                "panel the grey curve is the original signal and the blue curve is the denoised signal. The plots show "
                "that high-frequency fluctuations are reduced while the main waveform and physiologically meaningful peaks are retained.",
            )
        elif paragraph.text.startswith("Figure 5.1: Wavelet denoising"):
            replace_paragraph(paragraph, "Figure 5.1: Wavelet denoising quality check on one 30-second Sleep-EDF epoch. Grey: original signal; blue: denoised signal.")
        elif paragraph.text.startswith("The work can be extended in four directions"):
            replace_paragraph(paragraph, "The work can be extended in four directions: improving N1 recognition and temporal models, explaining predictions, generating evidence-based sleep reports, and validating the system on a second dataset.")
        elif "Complete the full-dataset experiments." in paragraph.text:
            replace_paragraph(paragraph, "2. Tune and validate temporal models. Optimise the CRF and BiLSTM using subject-wise validation, compare them on macro F1, Kappa and N1 F1, and test performance differences across subjects.")

    # Software environment was local; do not claim unverified GPU-backed experiments.
    replace_cell(document.tables[1].rows[7].cells[2], "Interactive development and local experiment execution")

    # Module status table reflects the actual feature-selection output and preliminary temporal comparison.
    replace_cell(document.tables[2].rows[4].cells[2], "Up to 67 selected features (64 in reported run)")
    replace_cell(document.tables[2].rows[5].cells[1], "Up to 67 selected features (64 in reported run)")
    replace_cell(document.tables[2].rows[6].cells[2], "Initial HMM, CRF and BiLSTM comparison")

    # Exact feature-family descriptions.
    replace_cell(
        document.tables[4].rows[1].cells[1],
        "Mean, variance, standard deviation, RMS, skewness, kurtosis, zero-crossing rate and peak-to-peak amplitude",
    )
    replace_cell(
        document.tables[4].rows[2].cells[1],
        "Absolute and relative delta, theta, alpha, sigma, beta and gamma power; spectral entropy; three band-power ratios",
    )
    replace_cell(document.tables[4].rows[3].cells[1], "Activity, mobility and complexity")

    # Completed testing table. T8 is clarified from generic resampling to class balancing.
    for row in (2, 3, 4, 5, 6, 7, 10):
        replace_cell(document.tables[6].rows[row].cells[4], "Done")
    replace_cell(document.tables[6].rows[6].cells[3], "Up to 67 features; 64 retained in the reported run")
    replace_cell(document.tables[6].rows[8].cells[2], "SMOTE and undersampling applied to training data only")
    replace_cell(document.tables[6].rows[8].cells[3], "Evaluation class distribution unchanged")
    replace_cell(document.tables[6].rows[8].cells[4], "Done")

    # Modality ablation had no stored N1-F1 breakdown; report the available Cohen's Kappa instead.
    ablation = document.tables[12]
    replace_cell(ablation.rows[0].cells[3], "Cohen's Kappa")
    values = [
        ("All modalities", "0.7584", "0.6963", "0.6722"),
        ("EEG only", "0.7037", "0.6381", "0.6014"),
        ("EOG only", "0.6524", "0.6027", "0.5396"),
        ("EMG only", "0.5285", "0.3791", "0.3320"),
    ]
    for row, values_row in zip(ablation.rows[1:], values):
        for cell, value in zip(row.cells, values_row):
            replace_cell(cell, value)

    # Update report dates in headers when they use an outdated future month.
    for section in document.sections:
        for paragraph in section.header.paragraphs:
            if "DECEMBER 2026" in paragraph.text:
                replace_paragraph(paragraph, paragraph.text.replace("DECEMBER 2026", "OCTOBER 2026"))

    clear_highlights(document)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
