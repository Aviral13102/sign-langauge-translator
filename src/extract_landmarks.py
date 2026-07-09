"""
extract_landmarks.py — Batch Landmark Extraction from Dataset Images
=====================================================================
Walks the ASL dataset directory structure (data/raw/<class>/images)
and extracts MediaPipe hand landmarks from each image, saving the
results as NumPy arrays for model training.

Usage:
    python src/extract_landmarks.py                    # Process full dataset
    python src/extract_landmarks.py --sample 100       # Process 100 images per class
    python src/extract_landmarks.py --data-dir path    # Custom dataset path
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks import HandLandmarkDetector


def extract_from_dataset(
    data_dir,
    output_dir,
    sample_per_class=None,
    min_detection_confidence=0.5,
):
    """
    Extract hand landmarks from all images in the dataset.

    The dataset is expected to have the structure:
        data_dir/
            A/
                image1.jpg
                image2.jpg
            B/
                ...
            ...

    Args:
        data_dir: Path to the raw dataset directory.
        output_dir: Path to save extracted landmark arrays.
        sample_per_class: If set, only process this many images per class.
        min_detection_confidence: MediaPipe detection confidence threshold.

    Returns:
        Dictionary with extraction statistics.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"[ERROR] Dataset directory not found: {data_path}")
        print(f"[INFO] Please download the ASL Alphabet dataset and place it in: {data_path}")
        print(f"[INFO] Expected structure: {data_path}/<A-Z>/<images>")
        return None

    # Initialise MediaPipe in static image mode (independent per-image processing)
    detector = HandLandmarkDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )

    import cv2

    all_landmarks = []
    all_labels = []
    stats = {
        "total_images": 0,
        "successful": 0,
        "failed": 0,
        "per_class": {},
    }

    # Get all class directories (A-Z, etc.)
    class_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir()],
        key=lambda x: x.name,
    )

    if len(class_dirs) == 0:
        print(f"[ERROR] No class subdirectories found in {data_path}")
        return None

    print(f"[INFO] Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    print(f"[INFO] Output directory: {output_path}")
    if sample_per_class:
        print(f"[INFO] Sampling {sample_per_class} images per class")
    print()

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])

        if sample_per_class and len(image_files) > sample_per_class:
            # Take evenly spaced samples
            step = len(image_files) // sample_per_class
            image_files = image_files[::step][:sample_per_class]

        class_success = 0
        class_fail = 0

        desc = f"[{class_name:>10}]"
        for img_path in tqdm(image_files, desc=desc, unit="img", ncols=80):
            stats["total_images"] += 1

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                class_fail += 1
                continue

            # Detect landmarks
            results = detector.detect(image)
            flat_landmarks = detector.extract_flat(results)

            if flat_landmarks is not None:
                all_landmarks.append(flat_landmarks)
                all_labels.append(class_name)
                class_success += 1
            else:
                class_fail += 1

        stats["per_class"][class_name] = {
            "total": len(image_files),
            "success": class_success,
            "failed": class_fail,
        }
        stats["successful"] += class_success
        stats["failed"] += class_fail

    detector.close()

    if len(all_landmarks) == 0:
        print("[ERROR] No landmarks extracted. Check your dataset.")
        return stats

    # Convert to NumPy arrays and save
    landmarks_array = np.array(all_landmarks, dtype=np.float32)
    labels_array = np.array(all_labels)

    landmarks_file = output_path / "landmarks.npy"
    labels_file = output_path / "labels.npy"

    np.save(str(landmarks_file), landmarks_array)
    np.save(str(labels_file), labels_array)

    print(f"\n{'='*60}")
    print(f"  Extraction Complete")
    print(f"{'='*60}")
    print(f"  Total images processed : {stats['total_images']}")
    print(f"  Successful extractions : {stats['successful']}")
    print(f"  Failed (no hand found) : {stats['failed']}")
    print(f"  Success rate           : {stats['successful']/max(stats['total_images'],1):.1%}")
    print(f"  Landmarks array shape  : {landmarks_array.shape}")
    print(f"  Labels array shape     : {labels_array.shape}")
    print(f"  Saved to               : {output_path}")
    print(f"{'='*60}")

    # Per-class breakdown
    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Class':<10} {'Total':>8} {'Success':>8} {'Failed':>8} {'Rate':>8}")
    print(f"  {'-'*44}")
    for class_name, class_stats in sorted(stats["per_class"].items()):
        rate = class_stats["success"] / max(class_stats["total"], 1)
        print(
            f"  {class_name:<10} {class_stats['total']:>8} "
            f"{class_stats['success']:>8} {class_stats['failed']:>8} "
            f"{rate:>7.1%}"
        )

    return stats


# --- CLI Entry Point ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from ASL dataset images."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to the raw dataset directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/landmarks",
        help="Path to save extracted landmarks (default: data/landmarks)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of images to sample per class (default: all)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="MediaPipe detection confidence threshold (default: 0.5)",
    )

    args = parser.parse_args()

    extract_from_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_per_class=args.sample,
        min_detection_confidence=args.confidence,
    )
