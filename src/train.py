"""
train.py -- MLP Classifier Training Pipeline
===============================================
Trains a scikit-learn MLPClassifier on normalised hand landmark
vectors (63 features) for ASL alphabet recognition (29 classes).

Pipeline:
  1. Load normalised landmark + label arrays
  2. Encode labels (LabelEncoder)
  3. Train/test split (stratified)
  4. Train MLPClassifier with early stopping
  5. Evaluate: classification report + confusion matrix
  6. Save model + label encoder with joblib
  7. Generate training plots

Usage:
    python src/train.py                                  # Train with defaults
    python src/train.py --augmented                      # Use augmented data
    python src/train.py --epochs 500 --hidden 256 128    # Custom architecture
"""

import sys
import argparse
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---- Default Configuration ------------------------------------------------

DEFAULT_LANDMARKS = "data/landmarks/landmarks_normalized.npy"
DEFAULT_LABELS = "data/landmarks/labels.npy"
MODEL_OUTPUT = "models/mlp_static.joblib"
ENCODER_OUTPUT = "models/label_encoder.joblib"
SCALER_OUTPUT = "models/scaler.joblib"
PLOTS_DIR = "models/plots"


def load_data(landmarks_path, labels_path):
    """
    Load landmark and label arrays.

    Returns:
        Tuple of (landmarks, labels) numpy arrays.
    """
    lm_path = Path(landmarks_path)
    lb_path = Path(labels_path)

    if not lm_path.exists():
        print(f"[ERROR] Landmarks file not found: {lm_path}")
        print("[INFO] Run extract_landmarks.py and normalize.py first.")
        sys.exit(1)

    if not lb_path.exists():
        print(f"[ERROR] Labels file not found: {lb_path}")
        sys.exit(1)

    landmarks = np.load(str(lm_path))
    labels = np.load(str(lb_path))

    print(f"[INFO] Loaded landmarks: {landmarks.shape}")
    print(f"[INFO] Loaded labels: {labels.shape}")
    print(f"[INFO] Unique classes: {len(np.unique(labels))}")
    print(f"[INFO] Classes: {sorted(np.unique(labels).tolist())}")

    return landmarks, labels


def train_model(
    landmarks_path=DEFAULT_LANDMARKS,
    labels_path=DEFAULT_LABELS,
    hidden_layers=(256, 128),
    max_epochs=300,
    learning_rate=0.001,
    batch_size=64,
    test_size=0.2,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    patience=15,
):
    """
    Full training pipeline.

    Args:
        landmarks_path: Path to normalised landmarks .npy file.
        labels_path: Path to labels .npy file.
        hidden_layers: Tuple defining MLP hidden layer sizes.
        max_epochs: Maximum training iterations.
        learning_rate: Initial learning rate for Adam optimizer.
        batch_size: Mini-batch size.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.
        early_stopping: Whether to use early stopping.
        validation_fraction: Fraction of training data for validation.
        patience: Number of iterations with no improvement before stopping.

    Returns:
        Trained model, label encoder, scaler, and test metrics dict.
    """
    print("=" * 62)
    print("  [ASL] MLP Classifier Training Pipeline")
    print("=" * 62)

    # ---- Step 1: Load data ----
    print("\n[Step 1/6] Loading data...")
    landmarks, labels = load_data(landmarks_path, labels_path)

    # ---- Step 2: Encode labels ----
    print("\n[Step 2/6] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"[INFO] Encoded {num_classes} classes: {label_encoder.classes_.tolist()}")

    # ---- Step 3: Train/test split ----
    print("\n[Step 3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        landmarks, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:     {X_test.shape[0]} samples")

    # ---- Step 3b: Scale features ----
    print("[INFO] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---- Step 4: Create and train MLP ----
    print(f"\n[Step 4/6] Training MLP classifier...")
    print(f"  Architecture       : {hidden_layers}")
    print(f"  Max epochs         : {max_epochs}")
    print(f"  Learning rate      : {learning_rate}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Early stopping     : {early_stopping}")
    if early_stopping:
        print(f"  Validation frac    : {validation_fraction}")
        print(f"  Patience           : {patience}")

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=max_epochs,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=patience,
        random_state=random_state,
        verbose=True,
    )

    start_time = time.time()
    mlp.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\n[INFO] Training complete in {train_time:.1f}s")
    print(f"[INFO] Iterations run: {mlp.n_iter_}")
    print(f"[INFO] Final loss: {mlp.loss_:.6f}")
    if hasattr(mlp, "best_loss_") and mlp.best_loss_ is not None:
        print(f"[INFO] Best validation loss: {mlp.best_loss_:.6f}")

    # ---- Step 5: Evaluate ----
    print(f"\n[Step 5/6] Evaluating model...")
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")

    # Classification report
    class_names = label_encoder.classes_.tolist()
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ---- Step 6: Save model ----
    print(f"\n[Step 6/6] Saving model and artifacts...")
    model_path = Path(MODEL_OUTPUT)
    encoder_path = Path(ENCODER_OUTPUT)
    scaler_path = Path(SCALER_OUTPUT)
    plots_path = Path(PLOTS_DIR)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(mlp, str(model_path))
    joblib.dump(label_encoder, str(encoder_path))
    joblib.dump(scaler, str(scaler_path))

    print(f"  [OK] Model saved:   {model_path}")
    print(f"  [OK] Encoder saved: {encoder_path}")
    print(f"  [OK] Scaler saved:  {scaler_path}")

    # ---- Generate plots ----
    print("[INFO] Generating training plots...")
    _plot_loss_curve(mlp, plots_path)
    _plot_confusion_matrix(cm, class_names, plots_path)
    _plot_per_class_accuracy(cm, class_names, plots_path)

    # ---- Final summary ----
    print(f"\n{'='*62}")
    print(f"  Training Summary")
    print(f"{'='*62}")
    print(f"  Test Accuracy      : {accuracy:.4f} ({accuracy:.1%})")
    print(f"  Training Time      : {train_time:.1f}s")
    print(f"  Iterations         : {mlp.n_iter_}")
    print(f"  Architecture       : {hidden_layers}")
    print(f"  Num Parameters     : {_count_params(mlp)}")
    print(f"  Model File         : {model_path}")
    print(f"  Model Size         : {model_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*62}")

    metrics = {
        "accuracy": accuracy,
        "train_time": train_time,
        "iterations": mlp.n_iter_,
        "final_loss": mlp.loss_,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return mlp, label_encoder, scaler, metrics


def _count_params(mlp):
    """Count total trainable parameters in the MLP."""
    total = 0
    for coef in mlp.coefs_:
        total += coef.size
    for intercept in mlp.intercepts_:
        total += intercept.size
    return total


def _plot_loss_curve(mlp, output_dir):
    """Plot and save the training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(mlp.loss_curve_, label="Training Loss", color="steelblue", linewidth=1.5)
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
        ax.plot(mlp.validation_scores_, label="Validation Accuracy",
                color="darkorange", linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss / Accuracy")
    ax.set_title("MLP Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = Path(output_dir) / "loss_curve.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Loss curve saved: {path}")


def _plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save the confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved: {path}")


def _plot_per_class_accuracy(cm, class_names, output_dir):
    """Plot per-class accuracy as a horizontal bar chart."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, max(8, len(class_names) * 0.35)))

    colors = ["#2ecc71" if acc >= 0.9 else "#f39c12" if acc >= 0.7 else "#e74c3c"
              for acc in per_class_acc]

    bars = ax.barh(range(len(class_names)), per_class_acc, color=colors, edgecolor="white")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=8)

    path = Path(output_dir) / "per_class_accuracy.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Per-class accuracy chart saved: {path}")


# ---- CLI Entry Point ------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP classifier for ASL sign language recognition."
    )
    parser.add_argument(
        "--landmarks",
        type=str,
        default=DEFAULT_LANDMARKS,
        help="Path to normalised landmarks .npy file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=DEFAULT_LABELS,
        help="Path to labels .npy file",
    )
    parser.add_argument(
        "--augmented",
        action="store_true",
        help="Use augmented data (data/landmarks/landmarks_augmented.npy)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer sizes (default: 256 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum training epochs (default: 300)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )

    args = parser.parse_args()

    # Override paths if using augmented data
    lm_path = args.landmarks
    lb_path = args.labels
    if args.augmented:
        lm_path = "data/landmarks/landmarks_augmented.npy"
        lb_path = "data/landmarks/labels_augmented.npy"
        print("[INFO] Using augmented dataset")

    train_model(
        landmarks_path=lm_path,
        labels_path=lb_path,
        hidden_layers=tuple(args.hidden),
        max_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        test_size=args.test_size,
        early_stopping=not args.no_early_stopping,
    )
