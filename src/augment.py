"""
augment.py -- Landmark Vector Augmentation
============================================
Applies data augmentation directly on normalised hand landmark
vectors (shape N x 63). This avoids expensive image-level transforms
and operates purely in landmark space.

Augmentation strategies:
  1. Gaussian noise     - adds small random noise to each coordinate
  2. Random scaling     - scales all landmarks by a random factor
  3. 2D rotation        - rotates x,y coordinates around the origin
  4. X-axis mirroring   - flips landmarks horizontally (left <-> right hand)

Usage:
    python src/augment.py                                    # Augment default files
    python src/augment.py --input data/landmarks/landmarks_normalized.npy
    python src/augment.py --multiplier 3                     # 3x augmented copies
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---- Individual Augmentation Functions ------------------------------------

def add_gaussian_noise(landmarks, std=0.01):
    """
    Add Gaussian noise to landmark coordinates.

    Args:
        landmarks: numpy array of shape (N, 63) or (63,).
        std: Standard deviation of the noise.

    Returns:
        Augmented landmarks with the same shape.
    """
    noise = np.random.normal(0, std, size=landmarks.shape).astype(landmarks.dtype)
    return landmarks + noise


def random_scale(landmarks, scale_range=(0.85, 1.15)):
    """
    Scale all landmark coordinates by a random factor.

    Args:
        landmarks: numpy array of shape (N, 63) or (63,).
        scale_range: Tuple of (min_scale, max_scale).

    Returns:
        Scaled landmarks.
    """
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return landmarks * factor


def rotate_2d(landmarks, max_angle_deg=15):
    """
    Apply a random 2D rotation to x,y coordinates.
    The z coordinate is left unchanged.

    Args:
        landmarks: numpy array of shape (N, 63) or (63,).
        max_angle_deg: Maximum rotation angle in degrees.

    Returns:
        Rotated landmarks.
    """
    single = landmarks.ndim == 1
    if single:
        landmarks = landmarks.reshape(1, -1)

    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    result = landmarks.copy()
    for i in range(len(result)):
        lm = result[i].reshape(21, 3)
        x_rot = lm[:, 0] * cos_a - lm[:, 1] * sin_a
        y_rot = lm[:, 0] * sin_a + lm[:, 1] * cos_a
        lm[:, 0] = x_rot
        lm[:, 1] = y_rot
        result[i] = lm.flatten()

    if single:
        result = result.flatten()
    return result


def mirror_x(landmarks):
    """
    Mirror landmarks along the X axis (horizontal flip).
    This simulates left-hand <-> right-hand conversion.

    Args:
        landmarks: numpy array of shape (N, 63) or (63,).

    Returns:
        Mirrored landmarks.
    """
    single = landmarks.ndim == 1
    if single:
        landmarks = landmarks.reshape(1, -1)

    result = landmarks.copy()
    for i in range(len(result)):
        lm = result[i].reshape(21, 3)
        lm[:, 0] = -lm[:, 0]  # negate x
        result[i] = lm.flatten()

    if single:
        result = result.flatten()
    return result


def augment_single(landmark_vec, noise_std=0.01, scale_range=(0.85, 1.15),
                   max_angle_deg=15, mirror_prob=0.3):
    """
    Apply a random combination of augmentations to a single vector.

    Args:
        landmark_vec: numpy array of shape (63,).
        noise_std: Gaussian noise standard deviation.
        scale_range: Scale range for random scaling.
        max_angle_deg: Max rotation angle.
        mirror_prob: Probability of applying mirroring.

    Returns:
        Augmented vector of shape (63,).
    """
    aug = landmark_vec.copy()

    # Always apply noise + scale + rotation
    aug = add_gaussian_noise(aug, std=noise_std)
    aug = random_scale(aug, scale_range=scale_range)
    aug = rotate_2d(aug, max_angle_deg=max_angle_deg)

    # Randomly mirror
    if np.random.random() < mirror_prob:
        aug = mirror_x(aug)

    return aug


# ---- Batch Augmentation --------------------------------------------------

def augment_dataset(landmarks, labels, multiplier=2, noise_std=0.01,
                    scale_range=(0.85, 1.15), max_angle_deg=15,
                    mirror_prob=0.3):
    """
    Augment an entire dataset by generating multiple augmented copies.

    Args:
        landmarks: numpy array of shape (N, 63).
        labels: numpy array of shape (N,).
        multiplier: Number of augmented copies per original sample.
        noise_std: Gaussian noise standard deviation.
        scale_range: Scale range.
        max_angle_deg: Max rotation angle.
        mirror_prob: Probability of mirroring.

    Returns:
        Tuple of (augmented_landmarks, augmented_labels) including originals.
        Shape: ((N * (1 + multiplier)), 63) and ((N * (1 + multiplier)),).
    """
    print(f"[INFO] Augmenting {len(landmarks)} samples with {multiplier}x multiplier...")
    print(f"[INFO] Noise std={noise_std}, Scale={scale_range}, "
          f"Rotation=+/-{max_angle_deg}deg, Mirror prob={mirror_prob}")

    aug_landmarks = [landmarks]  # start with originals
    aug_labels = [labels]

    for m in range(multiplier):
        batch = np.zeros_like(landmarks)
        for i in range(len(landmarks)):
            batch[i] = augment_single(
                landmarks[i],
                noise_std=noise_std,
                scale_range=scale_range,
                max_angle_deg=max_angle_deg,
                mirror_prob=mirror_prob,
            )
        aug_landmarks.append(batch)
        aug_labels.append(labels.copy())
        print(f"  [OK] Augmentation pass {m+1}/{multiplier} complete")

    combined_lm = np.concatenate(aug_landmarks, axis=0)
    combined_lb = np.concatenate(aug_labels, axis=0)

    # Shuffle
    indices = np.random.permutation(len(combined_lm))
    combined_lm = combined_lm[indices]
    combined_lb = combined_lb[indices]

    print(f"[INFO] Augmented dataset: {combined_lm.shape[0]} samples "
          f"(original: {len(landmarks)}, augmented: {combined_lm.shape[0] - len(landmarks)})")

    return combined_lm, combined_lb


# ---- CLI Entry Point ------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment normalised hand landmark data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/landmarks/landmarks_normalized.npy",
        help="Path to normalised landmarks .npy file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/landmarks/labels.npy",
        help="Path to labels .npy file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/landmarks/landmarks_augmented.npy",
        help="Path to save augmented landmarks",
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        default="data/landmarks/labels_augmented.npy",
        help="Path to save augmented labels",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="Number of augmented copies per sample (default: 2)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise standard deviation (default: 0.01)",
    )
    parser.add_argument(
        "--mirror-prob",
        type=float,
        default=0.3,
        help="Probability of mirroring (default: 0.3)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    labels_path = Path(args.labels)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        print("[INFO] Run normalize.py first to create normalised landmark files.")
        sys.exit(1)

    if not labels_path.exists():
        print(f"[ERROR] Labels file not found: {labels_path}")
        sys.exit(1)

    landmarks = np.load(str(input_path))
    labels = np.load(str(labels_path))

    print(f"[INFO] Loaded {landmarks.shape[0]} samples from {input_path}")

    aug_lm, aug_lb = augment_dataset(
        landmarks, labels,
        multiplier=args.multiplier,
        noise_std=args.noise_std,
        mirror_prob=args.mirror_prob,
    )

    output_path = Path(args.output)
    output_labels_path = Path(args.output_labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_path), aug_lm)
    np.save(str(output_labels_path), aug_lb)

    print(f"\n{'='*60}")
    print(f"  Augmentation Complete")
    print(f"{'='*60}")
    print(f"  Original samples   : {len(landmarks)}")
    print(f"  Augmented samples  : {len(aug_lm)}")
    print(f"  Multiplier         : {args.multiplier}")
    print(f"  Saved landmarks    : {output_path}")
    print(f"  Saved labels       : {output_labels_path}")
    print(f"{'='*60}")
