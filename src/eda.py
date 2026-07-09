"""
eda.py — Exploratory Data Analysis for ASL Landmarks
======================================================
Generates visual analyses of the extracted landmark dataset:
  - Class distribution bar chart
  - Missing/failed landmark statistics
  - Sample quality grid with overlaid landmarks
  - Feature distribution histograms
  - Train/test split preview

Usage:
    python src/eda.py                                 # Run full EDA
    python src/eda.py --landmarks data/landmarks/landmarks.npy
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter


def setup_plot_style():
    """Configure matplotlib for clean, publication-quality plots."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_class_distribution(labels, output_dir):
    """
    Bar chart showing the number of samples per ASL class.

    Args:
        labels: numpy array of class labels.
        output_dir: Directory to save the plot.
    """
    counts = Counter(labels)
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(classes, values, color=sns.color_palette("viridis", len(classes)),
                  edgecolor="white", linewidth=0.5)

    ax.set_xlabel("ASL Letter / Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution — ASL Landmark Dataset")

    # Add count labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(val), ha="center", va="bottom", fontsize=8,
        )

    # Add mean line
    mean_count = np.mean(values)
    ax.axhline(y=mean_count, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_count:.0f}")
    ax.legend()

    plt.tight_layout()
    save_path = Path(output_dir) / "class_distribution.png"
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Class distribution plot saved: {save_path}")

    return counts


def plot_feature_distributions(landmarks, labels, output_dir):
    """
    Histograms and boxplots of landmark coordinate distributions.

    Args:
        landmarks: numpy array of shape (N, 63).
        labels: numpy array of class labels.
        output_dir: Directory to save plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Split into x, y, z coordinates
    coord_names = ["X", "Y", "Z"]
    for idx, (ax, name) in enumerate(zip(axes, coord_names)):
        # Extract every 3rd value starting at idx (all x's, all y's, or all z's)
        coord_values = landmarks[:, idx::3].flatten()

        ax.hist(coord_values, bins=100, alpha=0.7, color=sns.color_palette("viridis")[idx],
                edgecolor="none")
        ax.set_title(f"{name} Coordinate Distribution")
        ax.set_xlabel(f"{name} Value")
        ax.set_ylabel("Frequency")
        ax.axvline(x=np.mean(coord_values), color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {np.mean(coord_values):.3f}")
        ax.legend(fontsize=9)

    plt.suptitle("Landmark Coordinate Distributions (Raw)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = Path(output_dir) / "feature_distributions.png"
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Feature distribution plot saved: {save_path}")


def plot_normalized_distributions(landmarks_norm, output_dir):
    """
    Distribution plots for normalised landmarks to verify normalization.

    Args:
        landmarks_norm: numpy array of shape (N, 63) — normalised.
        output_dir: Directory to save plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    coord_names = ["X (norm)", "Y (norm)", "Z (norm)"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for idx, (ax, name) in enumerate(zip(axes, coord_names)):
        coord_values = landmarks_norm[:, idx::3].flatten()

        ax.hist(coord_values, bins=100, alpha=0.7, color=colors[idx], edgecolor="none")
        ax.set_title(f"{name} Distribution")
        ax.set_xlabel(f"{name} Value")
        ax.set_ylabel("Frequency")

        # Check that wrist values are ~0
        wrist_val = np.mean(landmarks_norm[:, idx])
        ax.axvline(x=wrist_val, color="black", linestyle="--", alpha=0.7,
                   label=f"Wrist mean: {wrist_val:.4f}")
        ax.legend(fontsize=9)

    plt.suptitle("Normalised Landmark Distributions (Wrist-Relative)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = Path(output_dir) / "normalized_distributions.png"
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Normalised distribution plot saved: {save_path}")


def plot_sample_landmarks(landmarks, labels, output_dir, samples_per_class=1):
    """
    Visualise landmark positions for a sample from each class.

    Args:
        landmarks: numpy array of shape (N, 63).
        labels: numpy array of class labels.
        output_dir: Directory to save plots.
        samples_per_class: Number of samples to plot per class.
    """
    unique_classes = sorted(np.unique(labels))
    n_classes = len(unique_classes)

    cols = min(9, n_classes)
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else axes.tolist()
        axes = [axes] if isinstance(axes[0], plt.Axes) else axes
    axes_flat = [ax for row in axes for ax in (row if isinstance(row, (list, np.ndarray)) else [row])]

    # MediaPipe hand connections for drawing bones
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17),             # Palm
    ]

    for idx, class_name in enumerate(unique_classes):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]

        # Get a sample for this class
        class_mask = labels == class_name
        class_landmarks = landmarks[class_mask]
        if len(class_landmarks) == 0:
            ax.set_title(f"{class_name} (no data)")
            ax.axis("off")
            continue

        sample = class_landmarks[0].reshape(21, 3)

        # Plot connections (bones)
        for start, end in connections:
            ax.plot(
                [sample[start, 0], sample[end, 0]],
                [sample[start, 1], sample[end, 1]],
                "g-", alpha=0.4, linewidth=1,
            )

        # Plot landmarks
        ax.scatter(sample[:, 0], sample[:, 1], c="green", s=20, zorder=5)
        ax.scatter(sample[0, 0], sample[0, 1], c="red", s=40, zorder=6, marker="x")  # Wrist

        ax.set_title(f"'{class_name}'", fontsize=11, fontweight="bold")
        ax.invert_yaxis()  # Match image coordinate system
        ax.set_aspect("equal")
        ax.axis("off")

    # Hide unused axes
    for idx in range(len(unique_classes), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle("Sample Landmarks per Class (Green = joints, Red X = wrist)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = Path(output_dir) / "sample_landmarks.png"
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sample landmarks plot saved: {save_path}")


def plot_train_test_split(labels, output_dir, test_size=0.2):
    """
    Preview a stratified train/test split and verify class balance.

    Args:
        labels: numpy array of class labels.
        output_dir: Directory to save the plot.
        test_size: Fraction for the test set.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42
    )

    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    classes = sorted(set(labels))
    train_vals = [train_counts.get(c, 0) for c in classes]
    test_vals = [test_counts.get(c, 0) for c in classes]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width / 2, train_vals, width, label=f"Train ({len(train_idx)})", color="#2ecc71")
    ax.bar(x + width / 2, test_vals, width, label=f"Test ({len(test_idx)})", color="#e74c3c")

    ax.set_xlabel("ASL Letter / Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Stratified Train/Test Split ({1-test_size:.0%}/{test_size:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()
    save_path = Path(output_dir) / "train_test_split.png"
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Train/test split plot saved: {save_path}")


def compute_statistics(landmarks, labels):
    """
    Compute and print summary statistics for the dataset.

    Args:
        landmarks: numpy array of shape (N, 63).
        labels: numpy array of class labels.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset Summary Statistics")
    print(f"{'='*60}")
    print(f"  Total samples     : {len(labels)}")
    print(f"  Feature dimensions: {landmarks.shape[1]}")
    print(f"  Unique classes    : {len(np.unique(labels))}")
    print(f"  Classes           : {sorted(np.unique(labels).tolist())}")
    print()

    counts = Counter(labels)
    vals = list(counts.values())
    print(f"  Samples per class:")
    print(f"    Min   : {min(vals)} ({min(counts, key=counts.get)})")
    print(f"    Max   : {max(vals)} ({max(counts, key=counts.get)})")
    print(f"    Mean  : {np.mean(vals):.1f}")
    print(f"    Std   : {np.std(vals):.1f}")
    print()

    print(f"  Feature value ranges (raw):")
    print(f"    X: [{landmarks[:, 0::3].min():.4f}, {landmarks[:, 0::3].max():.4f}]")
    print(f"    Y: [{landmarks[:, 1::3].min():.4f}, {landmarks[:, 1::3].max():.4f}]")
    print(f"    Z: [{landmarks[:, 2::3].min():.4f}, {landmarks[:, 2::3].max():.4f}]")

    # Check for any NaN or Inf values
    nan_count = np.sum(np.isnan(landmarks))
    inf_count = np.sum(np.isinf(landmarks))
    print(f"\n  Data quality:")
    print(f"    NaN values : {nan_count}")
    print(f"    Inf values : {inf_count}")
    print(f"{'='*60}")


def run_eda(landmarks_path, labels_path, normalized_path=None, output_dir="data/eda_plots"):
    """
    Run the full EDA pipeline.

    Args:
        landmarks_path: Path to raw landmarks .npy file.
        labels_path: Path to labels .npy file.
        normalized_path: Path to normalised landmarks .npy file (optional).
        output_dir: Directory to save all plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    # Load data
    print(f"[INFO] Loading data...")
    landmarks_file = Path(landmarks_path)
    labels_file = Path(labels_path)

    if not landmarks_file.exists():
        print(f"[ERROR] Landmarks file not found: {landmarks_file}")
        print(f"[INFO] Run extract_landmarks.py first.")
        return

    if not labels_file.exists():
        print(f"[ERROR] Labels file not found: {labels_file}")
        return

    landmarks = np.load(str(landmarks_file))
    labels = np.load(str(labels_file))

    print(f"[INFO] Loaded {len(labels)} samples with {landmarks.shape[1]} features")
    print(f"[INFO] Generating plots in: {output_path}\n")

    # 1. Summary statistics
    compute_statistics(landmarks, labels)

    # 2. Class distribution
    print(f"\n[INFO] Generating plots...")
    plot_class_distribution(labels, output_path)

    # 3. Feature distributions
    plot_feature_distributions(landmarks, labels, output_path)

    # 4. Sample landmarks
    plot_sample_landmarks(landmarks, labels, output_path)

    # 5. Train/test split
    plot_train_test_split(labels, output_path)

    # 6. Normalised distributions (if available)
    norm_path = Path(normalized_path) if normalized_path else Path("data/landmarks/landmarks_normalized.npy")
    if norm_path.exists():
        landmarks_norm = np.load(str(norm_path))
        plot_normalized_distributions(landmarks_norm, output_path)
    else:
        print(f"  ⓘ Normalised landmarks not found. Run normalize.py to generate.")

    print(f"\n[INFO] EDA complete! All plots saved to: {output_path}")


# ─── CLI Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis on ASL landmark dataset."
    )
    parser.add_argument(
        "--landmarks",
        type=str,
        default="data/landmarks/landmarks.npy",
        help="Path to raw landmarks .npy file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/landmarks/labels.npy",
        help="Path to labels .npy file",
    )
    parser.add_argument(
        "--normalized",
        type=str,
        default=None,
        help="Path to normalised landmarks .npy file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eda_plots",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    run_eda(
        landmarks_path=args.landmarks,
        labels_path=args.labels,
        normalized_path=args.normalized,
        output_dir=args.output_dir,
    )
