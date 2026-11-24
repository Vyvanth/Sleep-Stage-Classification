"""Generate a report-ready system architecture diagram."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path("outputs/system_architecture_diagram.png")


def box(ax, x, y, width, height, title, body, color):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4, edgecolor="#334155", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height - 0.042, title, ha="center", va="top", fontsize=11,
            fontweight="bold", color="#0f172a")
    ax.text(x + width / 2, y + height / 2 - 0.022, body, ha="center", va="center", fontsize=8.7,
            color="#1e293b", linespacing=1.45)


def arrow(ax, start, end, label=None):
    line = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.5,
                           color="#475569", connectionstyle="arc3,rad=0")
    ax.add_patch(line)
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2
        ax.text(x, y + 0.018, label, ha="center", va="bottom", fontsize=7.5, color="#475569")


def main():
    fig, ax = plt.subplots(figsize=(15, 9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.963, "Multimodal Sleep-Stage Classification System Architecture", ha="center",
            va="center", fontsize=18, fontweight="bold", color="#0f172a")
    ax.text(0.5, 0.93, "Sleep-EDF processing, subject-wise learning, temporal modelling, and explainable outputs",
            ha="center", va="center", fontsize=10, color="#475569")

    box(ax, 0.04, 0.72, 0.19, 0.14, "Sleep-EDF Expanded", "Sleep Cassette (SC)\nSleep Telemetry (ST)\nPSG EDF + hypnogram EDF", "#e0f2fe")
    box(ax, 0.285, 0.72, 0.19, 0.14, "Signal Loading", "MNE-Python / pyEDFlib\nEEG Fpz-Cz and Pz-Oz\nEOG horizontal, EMG submental", "#dbeafe")
    box(ax, 0.53, 0.69, 0.20, 0.20, "Preprocessing", "Label mapping: W, N1, N2, N3, REM\nResample to 100 Hz\ndb4 wavelet denoising\nSubject-wise Z-score\n30-second epochs; wake trimming", "#dcfce7")
    box(ax, 0.78, 0.72, 0.18, 0.14, "Feature Dataset", "Per epoch: subject ID, record ID,\nepoch index, stage label\n116 candidate features", "#fef3c7")
    arrow(ax, (0.23, 0.79), (0.285, 0.79))
    arrow(ax, (0.475, 0.79), (0.53, 0.79))
    arrow(ax, (0.73, 0.79), (0.78, 0.79))

    box(ax, 0.14, 0.43, 0.22, 0.16, "Subject-wise Evaluation", "Subjects kept separate across\ntraining and test sets\nPrevents epoch-level data leakage", "#ede9fe")
    box(ax, 0.40, 0.43, 0.22, 0.16, "Ensemble Feature Selection", "LightGBM gain + mutual information\n+ permutation importance + stability\nModality weights; correlation filtering\nRetain up to 67 features", "#fce7f3")
    box(ax, 0.66, 0.43, 0.22, 0.16, "Epoch-wise Classifiers", "LightGBM and ExtraTrees\nClass weights or SMOTE-RUS\n5-stage probability estimates", "#ffedd5")
    # Route long links around the boxes so all text remains readable in reports.
    arrow(ax, (0.87, 0.72), (0.87, 0.64))
    arrow(ax, (0.87, 0.64), (0.25, 0.64))
    arrow(ax, (0.25, 0.64), (0.25, 0.59))
    arrow(ax, (0.36, 0.51), (0.40, 0.51))
    arrow(ax, (0.62, 0.51), (0.66, 0.51))

    box(ax, 0.12, 0.14, 0.22, 0.17, "Temporal Modelling", "Out-of-fold probabilities\nHMM / Viterbi smoothing\nLinear-chain CRF\nWindow BiLSTM", "#cffafe")
    box(ax, 0.40, 0.14, 0.22, 0.17, "Evaluation", "Accuracy, macro F1, Cohen's kappa\nPer-class precision, recall, F1\nN1 F1 emphasis\nConfusion matrices", "#e0e7ff")
    box(ax, 0.68, 0.14, 0.22, 0.17, "Explainable Outputs", "Selected-feature list\nTreeSHAP attributions\nEvidence-grounded\nclinical-style report", "#f3e8ff")
    arrow(ax, (0.77, 0.43), (0.77, 0.35))
    arrow(ax, (0.77, 0.35), (0.23, 0.35))
    arrow(ax, (0.23, 0.35), (0.23, 0.31))
    arrow(ax, (0.34, 0.225), (0.40, 0.225))
    arrow(ax, (0.62, 0.225), (0.68, 0.225))

    ax.text(0.5, 0.055, "All feature selection, sampling, tuning, and model fitting are performed on training subjects only.",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#334155")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
