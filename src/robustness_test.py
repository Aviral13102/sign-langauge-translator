"""
robustness_test.py -- Model Robustness Evaluation
====================================================
Tests the trained MLP model under various augmentation conditions
to assess real-world robustness: Gaussian noise, scale variation,
rotation around the wrist, and random landmark dropout.

Usage:
    python src/robustness_test.py
"""

import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---- Paths ----------------------------------------------------------------

MODEL_PATH = Path("models/mlp_static.joblib")
SCALER_PATH = Path("models/scaler.joblib")
ENCODER_PATH = Path("models/label_encoder.joblib")
LANDMARKS_PATH = Path("data/landmarks/landmarks_normalized.npy")
LABELS_PATH = Path("data/landmarks/labels.npy")
TEST_LM_PATH = Path("data/landmarks/test_landmarks.npy")
TEST_LB_PATH = Path("data/landmarks/test_labels.npy")
OUTPUT_PLOT = Path("models/plots/robustness_report.png")


# ---- Augmentation Functions -----------------------------------------------

def apply_gaussian_noise(X, sigma):
    """Add Gaussian noise with a given standard deviation."""
    noise = np.random.normal(0, sigma, size=X.shape).astype(X.dtype)
    return X + noise


def apply_scale(X, factor):
    """Uniformly scale all landmark coordinates."""
    return X * factor


def apply_rotation(X, angle_deg):
    """Rotate x,y coordinates around the wrist (index 0) by angle_deg."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    result = X.copy()
    for i in range(len(result)):
        lm = result[i].reshape(21, 3)
        # Wrist is already at origin after normalization
        x_rot = lm[:, 0] * cos_a - lm[:, 1] * sin_a
        y_rot = lm[:, 0] * sin_a + lm[:, 1] * cos_a
        lm[:, 0] = x_rot
        lm[:, 1] = y_rot
        result[i] = lm.flatten()
    return result


def apply_dropout(X, drop_fraction):
    """Zero out a random fraction of landmarks (entire x,y,z triplets)."""
    result = X.copy()
    num_landmarks = 21
    num_drop = max(1, int(num_landmarks * drop_fraction))
    for i in range(len(result)):
        lm = result[i].reshape(21, 3)
        drop_indices = np.random.choice(num_landmarks, size=num_drop, replace=False)
        lm[drop_indices] = 0.0
        result[i] = lm.flatten()
    return result


# ---- Evaluation Helper ----------------------------------------------------

def evaluate_augmented(model, scaler, X, y, augment_fn, param, param_name):
    """Apply augmentation and return accuracy."""
    X_aug = augment_fn(X, param)
    X_aug_scaled = scaler.transform(X_aug)
    y_pred = model.predict(X_aug_scaled)
    acc = accuracy_score(y, y_pred)
    return acc


# ---- Main -----------------------------------------------------------------

def main():
    print("=" * 66)
    print("  [ASL] MLP Robustness Test")
    print("=" * 66)
    print()

    # ---- Load model artifacts ----
    for fpath, name in [
        (MODEL_PATH, "Model"),
        (SCALER_PATH, "Scaler"),
        (ENCODER_PATH, "Label Encoder"),
    ]:
        if not fpath.exists():
            print(f"[ERROR] {name} not found: {fpath}")
            sys.exit(1)

    model = joblib.load(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    label_encoder = joblib.load(str(ENCODER_PATH))
    print(f"[INFO] Model loaded:   {MODEL_PATH}")
    print(f"[INFO] Scaler loaded:  {SCALER_PATH}")
    print(f"[INFO] Encoder loaded: {ENCODER_PATH}")

    # ---- Load test data ----
    if TEST_LM_PATH.exists() and TEST_LB_PATH.exists():
        print(f"[INFO] Loading dedicated test data...")
        X_test = np.load(str(TEST_LM_PATH))
        y_test_raw = np.load(str(TEST_LB_PATH))
        y_test = label_encoder.transform(y_test_raw)
    elif LANDMARKS_PATH.exists() and LABELS_PATH.exists():
        print(f"[INFO] Dedicated test files not found. Reproducing train/test split...")
        landmarks = np.load(str(LANDMARKS_PATH))
        labels = np.load(str(LABELS_PATH))
        y_encoded = label_encoder.transform(labels)
        _, X_test, _, y_test = train_test_split(
            landmarks, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded,
        )
    else:
        print("[ERROR] No landmark data found. Run the training pipeline first.")
        sys.exit(1)

    print(f"[INFO] Test samples: {X_test.shape[0]}")
    print(f"[INFO] Feature dim:  {X_test.shape[1]}")
    print(f"[INFO] Classes:      {len(label_encoder.classes_)}")
    print()

    # ---- Baseline accuracy ----
    X_test_scaled = scaler.transform(X_test)
    y_baseline = model.predict(X_test_scaled)
    baseline_acc = accuracy_score(y_test, y_baseline)
    print(f"[BASELINE] Clean test accuracy: {baseline_acc:.4f} ({baseline_acc:.1%})")
    print()

    # ---- Run robustness tests ----
    results = []  # list of (category, label, accuracy)

    # 1. Gaussian Noise
    print("-" * 66)
    print("  Test 1: Gaussian Noise")
    print("-" * 66)
    noise_sigmas = [0.01, 0.02, 0.05]
    for sigma in noise_sigmas:
        acc = evaluate_augmented(model, scaler, X_test, y_test,
                                 apply_gaussian_noise, sigma, "sigma")
        label = f"sigma={sigma}"
        results.append(("Gaussian Noise", label, acc))
        print(f"  sigma={sigma:<6} -> accuracy={acc:.4f} ({acc:.1%})")
    print()

    # 2. Scale Variation
    print("-" * 66)
    print("  Test 2: Scale Variation")
    print("-" * 66)
    scale_factors = [0.8, 0.9, 1.1, 1.2]
    for factor in scale_factors:
        acc = evaluate_augmented(model, scaler, X_test, y_test,
                                 apply_scale, factor, "scale")
        label = f"{factor}x"
        results.append(("Scale Variation", label, acc))
        print(f"  scale={factor:<6} -> accuracy={acc:.4f} ({acc:.1%})")
    print()

    # 3. Rotation (around wrist)
    print("-" * 66)
    print("  Test 3: Rotation (around wrist)")
    print("-" * 66)
    rotation_angles = [5, 10, 15, 20]
    for angle in rotation_angles:
        acc = evaluate_augmented(model, scaler, X_test, y_test,
                                 apply_rotation, angle, "degrees")
        label = f"{angle} deg"
        results.append(("Rotation", label, acc))
        print(f"  angle={angle:<4} deg -> accuracy={acc:.4f} ({acc:.1%})")
    print()

    # 4. Random Dropout (landmarks zeroed)
    print("-" * 66)
    print("  Test 4: Landmark Dropout")
    print("-" * 66)
    dropout_fracs = [0.05, 0.10]
    for frac in dropout_fracs:
        acc = evaluate_augmented(model, scaler, X_test, y_test,
                                 apply_dropout, frac, "dropout")
        label = f"{frac:.0%} drop"
        results.append(("Dropout", label, acc))
        pct = int(frac * 100)
        print(f"  dropout={pct}%    -> accuracy={acc:.4f} ({acc:.1%})")
    print()

    # ---- Summary Table ----
    print("=" * 66)
    print("  ROBUSTNESS SUMMARY")
    print("=" * 66)
    print()
    print(f"  {'Category':<20} {'Condition':<16} {'Accuracy':>10} {'Delta':>10}")
    print(f"  {'-'*20} {'-'*16} {'-'*10} {'-'*10}")
    print(f"  {'Baseline':<20} {'clean':<16} {baseline_acc:>9.1%} {'---':>10}")

    for category, label, acc in results:
        delta = acc - baseline_acc
        delta_str = f"{delta:+.1%}"
        print(f"  {category:<20} {label:<16} {acc:>9.1%} {delta_str:>10}")

    print()
    print("=" * 66)

    # ---- Bar chart ----
    print("[INFO] Generating robustness report chart...")
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    categories = {}
    for category, label, acc in results:
        if category not in categories:
            categories[category] = []
        categories[category].append((label, acc))

    num_categories = len(categories)
    fig, axes = plt.subplots(1, num_categories + 1, figsize=(4 * (num_categories + 1), 6),
                              gridspec_kw={"width_ratios": [1] + [len(v) for v in categories.values()]})

    # Baseline bar
    ax0 = axes[0]
    ax0.bar(["Baseline"], [baseline_acc], color="#2ecc71", edgecolor="white", width=0.5)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("Accuracy", fontsize=11)
    ax0.set_title("Baseline", fontsize=11, fontweight="bold")
    ax0.axhline(y=baseline_acc, color="gray", linestyle="--", alpha=0.5)
    ax0.text(0, baseline_acc + 0.02, f"{baseline_acc:.1%}", ha="center", fontsize=9, fontweight="bold")

    color_map = {
        "Gaussian Noise": "#3498db",
        "Scale Variation": "#e67e22",
        "Rotation": "#9b59b6",
        "Dropout": "#e74c3c",
    }

    for idx, (cat_name, items) in enumerate(categories.items(), start=1):
        ax = axes[idx]
        labels_list = [item[0] for item in items]
        accs = [item[1] for item in items]
        color = color_map.get(cat_name, "#34495e")

        bars = ax.bar(labels_list, accs, color=color, edgecolor="white", width=0.6, alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_title(cat_name, fontsize=11, fontweight="bold")
        ax.axhline(y=baseline_acc, color="#2ecc71", linestyle="--", alpha=0.6, label="Baseline")
        ax.tick_params(axis="x", rotation=30)

        for bar, acc_val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc_val + 0.02,
                    f"{acc_val:.1%}", ha="center", fontsize=8, fontweight="bold")

        if idx == 1:
            ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("MLP Robustness Under Augmentation Conditions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_PLOT), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Chart saved: {OUTPUT_PLOT}")
    print()
    print("[DONE] Robustness test complete.")
    print("=" * 66)


if __name__ == "__main__":
    main()
