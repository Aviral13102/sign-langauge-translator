"""
normalize.py — Wrist-Relative Landmark Normalization
======================================================
Normalises raw hand landmarks by subtracting the wrist position
(landmark index 0) from all 21 landmarks, making the feature
vectors position-invariant. Optionally applies scale normalization.

Usage:
    python src/normalize.py                           # Normalize saved landmarks
    python src/normalize.py --scale                   # Also apply scale normalization
"""

import argparse
import numpy as np
from pathlib import Path


def normalize_wrist_relative(landmarks):
    """
    Normalise landmarks relative to the wrist position.

    Subtracts the wrist (index 0) x, y, z from all 21 landmarks
    so the model becomes position-invariant.

    Args:
        landmarks: numpy array of shape (N, 63) — flat landmark vectors,
                   or (21, 3) for a single sample.

    Returns:
        Normalised numpy array of the same shape.
    """
    is_single = landmarks.ndim == 1
    if is_single:
        landmarks = landmarks.reshape(1, -1)

    normalized = landmarks.copy()

    for i in range(len(normalized)):
        # Reshape to (21, 3) for easier manipulation
        lm = normalized[i].reshape(21, 3)

        # Subtract wrist position from all landmarks
        wrist = lm[0].copy()
        lm = lm - wrist

        # Flatten back to (63,)
        normalized[i] = lm.flatten()

    if is_single:
        normalized = normalized.flatten()

    return normalized


def normalize_scale(landmarks):
    """
    Apply scale normalization so landmarks fit within a unit bounding box.

    After wrist-relative normalization, the landmark spread still depends
    on hand distance from the camera. Dividing by the maximum absolute
    value makes the features scale-invariant.

    Args:
        landmarks: numpy array of shape (N, 63) — wrist-normalised vectors.

    Returns:
        Scale-normalised numpy array of the same shape.
    """
    is_single = landmarks.ndim == 1
    if is_single:
        landmarks = landmarks.reshape(1, -1)

    normalized = landmarks.copy()

    for i in range(len(normalized)):
        max_val = np.max(np.abs(normalized[i]))
        if max_val > 0:
            normalized[i] = normalized[i] / max_val

    if is_single:
        normalized = normalized.flatten()

    return normalized


def normalize_single_frame(landmarks_21x3):
    """
    Convenience function for real-time inference.
    Takes (21, 3) landmarks and returns normalized (63,) vector.

    Args:
        landmarks_21x3: numpy array of shape (21, 3).

    Returns:
        Normalised flat vector of shape (63,).
    """
    flat = landmarks_21x3.flatten()
    normalized = normalize_wrist_relative(flat)
    normalized = normalize_scale(normalized)
    return normalized


def process_dataset(input_path, output_path, apply_scale=True):
    """
    Load raw landmarks, normalise, and save.

    Args:
        input_path: Path to the raw landmarks .npy file.
        output_path: Path to save the normalised landmarks.
        apply_scale: Whether to apply scale normalization in addition to wrist-relative.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        print(f"[INFO] Run extract_landmarks.py first to create landmark files.")
        return

    print(f"[INFO] Loading landmarks from: {input_file}")
    landmarks = np.load(str(input_file))
    print(f"[INFO] Loaded {landmarks.shape[0]} samples with {landmarks.shape[1]} features")

    # Step 1: Wrist-relative normalization
    print("[INFO] Applying wrist-relative normalization...")
    normalized = normalize_wrist_relative(landmarks)

    # Step 2: Optional scale normalization
    if apply_scale:
        print("[INFO] Applying scale normalization...")
        normalized = normalize_scale(normalized)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_file), normalized)

    print(f"\n{'='*60}")
    print(f"  Normalization Complete")
    print(f"{'='*60}")
    print(f"  Input shape       : {landmarks.shape}")
    print(f"  Output shape      : {normalized.shape}")
    print(f"  Wrist-relative    : Yes")
    print(f"  Scale normalized  : {'Yes' if apply_scale else 'No'}")
    print(f"  Saved to          : {output_file}")
    print(f"{'='*60}")

    # Quick sanity check
    print(f"\n  Sanity Check (first sample):")
    print(f"  Wrist landmark (should be ~0): {normalized[0][:3]}")
    print(f"  Value range: [{normalized.min():.4f}, {normalized.max():.4f}]")


# ─── CLI Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalise extracted hand landmarks."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/landmarks/landmarks.npy",
        help="Path to raw landmarks .npy file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/landmarks/landmarks_normalized.npy",
        help="Path to save normalised landmarks",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=True,
        help="Apply scale normalization (default: True)",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable scale normalization",
    )

    args = parser.parse_args()
    apply_scale = not args.no_scale

    process_dataset(
        input_path=args.input,
        output_path=args.output,
        apply_scale=apply_scale,
    )
