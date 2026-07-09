"""
evaluate.py -- Standalone Model Evaluation
=============================================
Loads a trained MLP model and runs comprehensive evaluation
on test data: confusion matrix, per-class metrics, most
confused pairs, and overall accuracy breakdown.

Usage:
    python src/evaluate.py
    python src/evaluate.py --model models/mlp_static.joblib
"""

import sys
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---- Default Paths --------------------------------------------------------

DEFAULT_MODEL = "models/mlp_static.joblib"
DEFAULT_ENCODER = "models/label_encoder.joblib"
DEFAULT_SCALER = "models/scaler.joblib"
DEFAULT_LANDMARKS = "data/landmarks/landmarks_normalized.npy"
DEFAULT_LABELS = "data/landmarks/labels.npy"
PLOTS_DIR = "models/plots"


def evaluate_model(
    model_path=DEFAULT_MODEL,
    encoder_path=DEFAULT_ENCODER,
    scaler_path=DEFAULT_SCALER,
    landmarks_path=DEFAULT_LANDMARKS,
    labels_path=DEFAULT_LABELS,
    test_size=0.2,
    random_state=42,
):
    """
    Run full model evaluation and generate reports.

    Args:
        model_path: Path to trained model .joblib file.
        encoder_path: Path to label encoder .joblib file.
        scaler_path: Path to scaler .joblib file.
        landmarks_path: Path to normalised landmarks .npy file.
        labels_path: Path to labels .npy file.
        test_size: Fraction for test set (must match training split).
        random_state: Random seed (must match training split).
    """
    print("=" * 62)
    print("  [ASL] Model Evaluation Report")
    print("=" * 62)

    # Load model + artifacts
    for fpath, name in [
        (model_path, "Model"),
        (encoder_path, "Label encoder"),
        (scaler_path, "Scaler"),
        (landmarks_path, "Landmarks"),
        (labels_path, "Labels"),
    ]:
        if not Path(fpath).exists():
            print(f"[ERROR] {name} not found: {fpath}")
            sys.exit(1)

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    landmarks = np.load(landmarks_path)
    labels = np.load(labels_path)

    print(f"[INFO] Model loaded from: {model_path}")
    print(f"[INFO] Dataset: {len(labels)} samples, {len(np.unique(labels))} classes")

    # Reproduce the same train/test split
    y_encoded = label_encoder.transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        landmarks, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    class_names = label_encoder.classes_.tolist()

    # ---- Overall Metrics ----
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    print(f"\n{'='*62}")
    print(f"  Overall Metrics")
    print(f"{'='*62}")
    print(f"  Accuracy           : {accuracy:.4f} ({accuracy:.1%})")
    print(f"  Macro F1           : {f1_macro:.4f}")
    print(f"  Weighted F1        : {f1_weighted:.4f}")
    print(f"  Macro Precision    : {precision:.4f}")
    print(f"  Macro Recall       : {recall:.4f}")
    print(f"  Test samples       : {len(y_test)}")

    # ---- Classification Report ----
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\n  Classification Report:\n{report}")

    # ---- Per-Class Accuracy ----
    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    print(f"\n  Per-Class Accuracy:")
    print(f"  {'Class':<12} {'Accuracy':>8} {'Correct':>8} {'Total':>8}")
    print(f"  {'-'*40}")

    weak_classes = []
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        total = cm[i].sum()
        correct = cm[i, i]
        marker = " *" if acc < 0.85 else ""
        print(f"  {name:<12} {acc:>7.1%} {correct:>8} {total:>8}{marker}")
        if acc < 0.85:
            weak_classes.append((name, acc))

    if weak_classes:
        print(f"\n  * Classes below 85% accuracy:")
        for name, acc in sorted(weak_classes, key=lambda x: x[1]):
            print(f"    - {name}: {acc:.1%}")

    # ---- Most Confused Pairs ----
    print(f"\n  Most Confused Pairs:")
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_cls, pred_cls, count in confused_pairs[:10]:
        print(f"    {true_cls:>8} -> {pred_cls:<8} : {count} misclassifications")

    # ---- Confidence Analysis ----
    max_proba = np.max(y_proba, axis=1)
    correct_mask = y_pred == y_test

    print(f"\n  Confidence Analysis:")
    for threshold in [0.5, 0.7, 0.85, 0.9, 0.95]:
        above = max_proba >= threshold
        if above.sum() > 0:
            acc_above = accuracy_score(y_test[above], y_pred[above])
            coverage = above.sum() / len(y_test)
            print(f"    Threshold {threshold:.0%}: "
                  f"accuracy={acc_above:.1%}, coverage={coverage:.1%} "
                  f"({above.sum()}/{len(y_test)} samples)")

    # ---- Save Plots ----
    plots_path = Path(PLOTS_DIR)
    plots_path.mkdir(parents=True, exist_ok=True)

    _plot_confidence_distribution(max_proba, correct_mask, plots_path)
    _plot_per_class_f1(y_test, y_pred, class_names, plots_path)

    print(f"\n[INFO] Evaluation complete. Plots saved to: {plots_path}")
    print("=" * 62)


def _plot_confidence_distribution(max_proba, correct_mask, output_dir):
    """Plot confidence distribution for correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(max_proba[correct_mask], bins=50, alpha=0.7, color="#2ecc71",
            label=f"Correct ({correct_mask.sum()})", edgecolor="none")
    ax.hist(max_proba[~correct_mask], bins=50, alpha=0.7, color="#e74c3c",
            label=f"Incorrect ({(~correct_mask).sum()})", edgecolor="none")

    ax.axvline(x=0.85, color="black", linestyle="--", alpha=0.7, label="85% threshold")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = Path(output_dir) / "confidence_distribution.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Confidence distribution saved: {path}")


def _plot_per_class_f1(y_test, y_pred, class_names, output_dir):
    """Plot per-class F1 scores."""
    f1_per_class = f1_score(y_test, y_pred, average=None)

    fig, ax = plt.subplots(figsize=(10, max(8, len(class_names) * 0.35)))

    colors = ["#2ecc71" if f1 >= 0.9 else "#f39c12" if f1 >= 0.7 else "#e74c3c"
              for f1 in f1_per_class]

    ax.barh(range(len(class_names)), f1_per_class, color=colors, edgecolor="white")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    for i, f1 in enumerate(f1_per_class):
        ax.text(f1 + 0.01, i, f"{f1:.1%}", va="center", fontsize=8)

    path = Path(output_dir) / "per_class_f1.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Per-class F1 chart saved: {path}")


# ---- CLI Entry Point -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained ASL MLP classifier."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Label encoder path")
    parser.add_argument("--scaler", default=DEFAULT_SCALER, help="Scaler path")
    parser.add_argument("--landmarks", default=DEFAULT_LANDMARKS, help="Landmarks path")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Labels path")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        encoder_path=args.encoder,
        scaler_path=args.scaler,
        landmarks_path=args.landmarks,
        labels_path=args.labels,
    )
