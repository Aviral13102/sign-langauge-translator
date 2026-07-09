"""
collect_data.py -- Webcam-Based ASL Data Collector
====================================================
Records hand landmark vectors through the webcam for each
ASL sign class. Provides a guided UI with countdown timer,
per-class progress bar, and automatic saving.

Usage:
    python src/collect_data.py                         # Collect all 26 letters
    python src/collect_data.py --classes A B C         # Collect specific letters
    python src/collect_data.py --samples 100           # 100 samples per class
    python src/collect_data.py --output data/custom    # Custom output directory
"""

import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture import WebcamCapture
from src.landmarks import HandLandmarkDetector
from src.normalize import normalize_single_frame


# ---- Default Configuration ------------------------------------------------

DEFAULT_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
DEFAULT_SAMPLES_PER_CLASS = 200
DEFAULT_OUTPUT_DIR = "data/custom"
COUNTDOWN_SECONDS = 3
COLLECTION_DELAY_MS = 80  # minimum ms between samples (avoid duplicates)


def draw_collection_ui(frame, class_name, collected, target, phase, countdown=None):
    """
    Overlay the data collection UI on the frame.

    Args:
        frame: BGR image.
        class_name: Current class being collected.
        collected: Number of samples collected so far.
        target: Target number of samples.
        phase: 'waiting', 'countdown', or 'recording'.
        countdown: Countdown seconds remaining (for countdown phase).
    """
    h, w, _ = frame.shape

    # -- Top bar with current class --
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(
        frame, f"Collecting: '{class_name}'", (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA,
    )

    # -- Progress bar --
    bar_x, bar_y, bar_w, bar_h = 15, 70, w - 30, 25
    progress = min(collected / max(target, 1), 1.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * progress)
    bar_color = (0, 200, 0) if progress < 1.0 else (0, 255, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    progress_text = f"{collected}/{target} ({progress:.0%})"
    cv2.putText(
        frame, progress_text, (bar_x + 5, bar_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # -- Phase-specific display --
    if phase == "waiting":
        msg = "Press SPACE to start recording"
        cv2.putText(
            frame, msg, (w // 2 - 250, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
    elif phase == "countdown" and countdown is not None:
        cv2.putText(
            frame, str(countdown), (w // 2 - 30, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 6, cv2.LINE_AA,
        )
    elif phase == "recording":
        # Flashing red dot
        if int(time.time() * 4) % 2 == 0:
            cv2.circle(frame, (w - 40, 35), 12, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (w - 90, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    # -- Bottom instructions --
    instructions = "SPACE=Start/Pause | S=Skip class | Q=Quit & save"
    cv2.rectangle(frame, (0, h - 35), (w, h), (40, 40, 40), -1)
    cv2.putText(
        frame, instructions, (15, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )


def collect_class_data(
    cam, detector, class_name, target_samples, window_name
):
    """
    Collect landmark samples for a single class.

    Returns:
        List of normalised landmark vectors (each shape (63,)),
        or None if the user chose to quit entirely.
    """
    samples = []
    phase = "waiting"  # waiting -> countdown -> recording
    countdown_start = None
    last_sample_time = 0

    for success, frame in cam.get_frames():
        results = detector.detect(frame)
        detector.draw_landmarks(frame, results)
        cam.draw_fps(frame)

        # Determine countdown value
        countdown_val = None
        if phase == "countdown" and countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            if remaining <= 0:
                phase = "recording"
            else:
                countdown_val = remaining

        draw_collection_ui(
            frame, class_name, len(samples), target_samples, phase, countdown_val
        )
        cv2.imshow(window_name, frame)

        # -- Collect samples when recording --
        if phase == "recording":
            now_ms = time.time() * 1000
            if now_ms - last_sample_time >= COLLECTION_DELAY_MS:
                landmarks = detector.extract_landmarks(results)
                if landmarks is not None:
                    normalized = normalize_single_frame(landmarks)
                    samples.append(normalized)
                    last_sample_time = now_ms

            if len(samples) >= target_samples:
                print(f"  [OK] Collected {len(samples)} samples for '{class_name}'")
                return samples

        # -- Handle keypresses --
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None  # signal to quit
        elif key == ord(" "):
            if phase == "waiting" or phase == "recording":
                if phase == "waiting":
                    phase = "countdown"
                    countdown_start = time.time()
                    print(f"  [INFO] Countdown started for '{class_name}'...")
                else:
                    # Pause
                    phase = "waiting"
                    print(f"  [INFO] Paused at {len(samples)}/{target_samples}")
            elif phase == "countdown":
                phase = "waiting"
        elif key == ord("s"):
            print(f"  [INFO] Skipped class '{class_name}' ({len(samples)} samples collected)")
            return samples  # return whatever we have


def run_collector(
    classes=None,
    samples_per_class=DEFAULT_SAMPLES_PER_CLASS,
    output_dir=DEFAULT_OUTPUT_DIR,
    camera_index=0,
):
    """
    Main data collection loop.

    Args:
        classes: List of class labels to collect.
        samples_per_class: Number of samples to record per class.
        output_dir: Directory to save .npy output files.
        camera_index: Camera device index.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    window_name = "ASL Data Collector"

    print("=" * 62)
    print("  [ASL] Webcam Data Collector")
    print("=" * 62)
    print(f"  Classes            : {len(classes)}")
    print(f"  Samples per class  : {samples_per_class}")
    print(f"  Output directory   : {output_path}")
    print(f"  Camera index       : {camera_index}")
    print("=" * 62)
    print()

    cam = WebcamCapture(camera_index=camera_index, window_name=window_name)
    detector = HandLandmarkDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )
    cam.start()

    all_landmarks = []
    all_labels = []
    quit_requested = False

    try:
        for i, class_name in enumerate(classes):
            print(f"\n  [{i+1}/{len(classes)}] Class: '{class_name}'")

            samples = collect_class_data(
                cam, detector, class_name, samples_per_class, window_name
            )

            if samples is None:
                # User pressed Q
                quit_requested = True
                print("\n[INFO] Quit requested. Saving collected data...")
                break

            if len(samples) > 0:
                for s in samples:
                    all_landmarks.append(s)
                    all_labels.append(class_name)

        # ---- Save collected data ----
        if len(all_landmarks) > 0:
            landmarks_array = np.array(all_landmarks, dtype=np.float32)
            labels_array = np.array(all_labels)

            lm_file = output_path / "landmarks_custom.npy"
            lb_file = output_path / "labels_custom.npy"

            np.save(str(lm_file), landmarks_array)
            np.save(str(lb_file), labels_array)

            print(f"\n{'='*62}")
            print(f"  Collection Complete")
            print(f"{'='*62}")
            print(f"  Total samples      : {len(all_landmarks)}")
            print(f"  Unique classes     : {len(set(all_labels))}")
            print(f"  Landmarks shape    : {landmarks_array.shape}")
            print(f"  Saved landmarks    : {lm_file}")
            print(f"  Saved labels       : {lb_file}")
            print(f"{'='*62}")

            # Per-class summary
            from collections import Counter
            counts = Counter(all_labels)
            print(f"\n  Per-Class Summary:")
            print(f"  {'Class':<10} {'Samples':>10}")
            print(f"  {'-'*22}")
            for cls in sorted(counts.keys()):
                print(f"  {cls:<10} {counts[cls]:>10}")
        else:
            print("\n[WARNING] No samples collected. Nothing saved.")

    finally:
        detector.close()
        cam.stop()
        if not quit_requested:
            print("\n[INFO] All classes complete. Goodbye!")



# ---- Sequence Recording Mode (for LSTM) -----------------------------------

DEFAULT_GESTURE_CLASSES = [
    "hello", "thank_you", "yes", "no", "please",
    "sorry", "help", "goodbye", "i_love_you", "stop",
]
SEQUENCE_LENGTH = 30  # 30 frames = ~1 second at 30fps
DEFAULT_SEQUENCES_PER_CLASS = 50
DEFAULT_SEQUENCE_OUTPUT = "data/sequences"


def draw_sequence_ui(frame, gesture_name, collected, target, phase, countdown=None,
                     frame_count=None, seq_length=SEQUENCE_LENGTH):
    """Overlay the sequence recording UI on the frame."""
    h, w, _ = frame.shape

    # -- Top bar --
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(
        frame, f"Gesture: '{gesture_name}'", (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 2, cv2.LINE_AA,
    )

    # -- Sequence progress bar --
    bar_x, bar_y, bar_w, bar_h = 15, 70, w - 30, 25
    progress = min(collected / max(target, 1), 1.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * progress)
    bar_color = (0, 200, 0) if progress < 1.0 else (0, 255, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    progress_text = f"Sequences: {collected}/{target} ({progress:.0%})"
    cv2.putText(
        frame, progress_text, (bar_x + 5, bar_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # -- Phase display --
    if phase == "waiting":
        msg = "Press SPACE to record a sequence"
        cv2.putText(
            frame, msg, (w // 2 - 270, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
    elif phase == "countdown" and countdown is not None:
        cv2.putText(
            frame, str(countdown), (w // 2 - 30, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 6, cv2.LINE_AA,
        )
    elif phase == "recording" and frame_count is not None:
        # Frame capture progress
        seq_progress = frame_count / seq_length
        rec_bar_w = w - 30
        rec_bar_y = h // 2 + 60
        cv2.rectangle(frame, (15, rec_bar_y), (15 + rec_bar_w, rec_bar_y + 20), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, rec_bar_y), (15 + int(rec_bar_w * seq_progress), rec_bar_y + 20),
                      (0, 0, 255), -1)
        cv2.putText(
            frame, f"Recording: {frame_count}/{seq_length}", (15, rec_bar_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )
        # Flashing REC dot
        if int(time.time() * 4) % 2 == 0:
            cv2.circle(frame, (w - 40, 35), 12, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (w - 90, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    # -- Bottom instructions --
    instructions = "SPACE=Record | S=Skip gesture | Q=Quit & save"
    cv2.rectangle(frame, (0, h - 35), (w, h), (40, 40, 40), -1)
    cv2.putText(
        frame, instructions, (15, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )


def collect_gesture_sequences(
    cam, detector, gesture_name, target_sequences, window_name
):
    """
    Collect 30-frame sequences for a single gesture.

    Returns:
        List of sequences (each shape (30, 63)), or None if user wants to quit.
    """
    sequences = []
    phase = "waiting"
    countdown_start = None
    current_sequence = []

    for success, frame in cam.get_frames():
        results = detector.detect(frame)
        detector.draw_landmarks(frame, results)
        cam.draw_fps(frame)

        # Handle countdown
        countdown_val = None
        if phase == "countdown" and countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            if remaining <= 0:
                phase = "recording"
                current_sequence = []
                print(f"    Recording sequence {len(sequences)+1}...")
            else:
                countdown_val = remaining

        draw_sequence_ui(
            frame, gesture_name, len(sequences), target_sequences,
            phase, countdown_val,
            frame_count=len(current_sequence) if phase == "recording" else None,
        )
        cv2.imshow(window_name, frame)

        # -- Record frames --
        if phase == "recording":
            landmarks = detector.extract_landmarks(results)
            if landmarks is not None:
                normalized = normalize_single_frame(landmarks)
                current_sequence.append(normalized)

                if len(current_sequence) >= SEQUENCE_LENGTH:
                    # Sequence complete
                    sequences.append(np.array(current_sequence, dtype=np.float32))
                    current_sequence = []
                    phase = "waiting"
                    print(f"    [OK] Sequence {len(sequences)}/{target_sequences} recorded")

                    if len(sequences) >= target_sequences:
                        return sequences

        # -- Keypresses --
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        elif key == ord(" ") and phase == "waiting":
            phase = "countdown"
            countdown_start = time.time()
        elif key == ord("s"):
            print(f"    [INFO] Skipped '{gesture_name}' ({len(sequences)} sequences)")
            return sequences


def run_sequence_collector(
    gestures=None,
    sequences_per_class=DEFAULT_SEQUENCES_PER_CLASS,
    output_dir=DEFAULT_SEQUENCE_OUTPUT,
    camera_index=0,
):
    """
    Collect 30-frame gesture sequences for LSTM training.

    Args:
        gestures: List of gesture names to collect.
        sequences_per_class: Number of sequences per gesture.
        output_dir: Directory to save output.
        camera_index: Camera device index.
    """
    if gestures is None:
        gestures = DEFAULT_GESTURE_CLASSES

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    window_name = "ASL Gesture Sequence Recorder"

    print("=" * 62)
    print("  [ASL] Dynamic Gesture Sequence Recorder")
    print("=" * 62)
    print(f"  Gestures           : {len(gestures)}")
    print(f"  Sequences/gesture  : {sequences_per_class}")
    print(f"  Frames/sequence    : {SEQUENCE_LENGTH}")
    print(f"  Output directory   : {output_path}")
    print("=" * 62)
    print()

    cam = WebcamCapture(camera_index=camera_index, window_name=window_name)
    detector = HandLandmarkDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )
    cam.start()

    all_sequences = []
    all_labels = []
    quit_requested = False

    try:
        for i, gesture in enumerate(gestures):
            print(f"\n  [{i+1}/{len(gestures)}] Gesture: '{gesture}'")

            seqs = collect_gesture_sequences(
                cam, detector, gesture, sequences_per_class, window_name
            )

            if seqs is None:
                quit_requested = True
                print("\n[INFO] Quit requested. Saving collected data...")
                break

            for s in seqs:
                all_sequences.append(s)
                all_labels.append(gesture)

        # ---- Save ----
        if len(all_sequences) > 0:
            seq_array = np.array(all_sequences, dtype=np.float32)
            lbl_array = np.array(all_labels)

            seq_file = output_path / "sequences.npy"
            lbl_file = output_path / "labels.npy"

            np.save(str(seq_file), seq_array)
            np.save(str(lbl_file), lbl_array)

            print(f"\n{'='*62}")
            print(f"  Sequence Collection Complete")
            print(f"{'='*62}")
            print(f"  Total sequences    : {len(all_sequences)}")
            print(f"  Sequence shape     : {seq_array.shape}")
            print(f"  Unique gestures    : {len(set(all_labels))}")
            print(f"  Saved to           : {output_path}")
            print(f"{'='*62}")
        else:
            print("\n[WARNING] No sequences collected. Nothing saved.")

    finally:
        detector.close()
        cam.stop()


# ---- CLI Entry Point ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Webcam-based ASL data collector with guided UI."
    )
    parser.add_argument(
        "--mode",
        choices=["static", "sequence"],
        default="static",
        help="Collection mode: 'static' for single frames (MLP), 'sequence' for 30-frame sequences (LSTM)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific classes/gestures to collect",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=f"Number of samples per class (default: {DEFAULT_SAMPLES_PER_CLASS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )

    args = parser.parse_args()

    if args.mode == "sequence":
        run_sequence_collector(
            gestures=args.classes,
            sequences_per_class=args.samples if args.samples != DEFAULT_SAMPLES_PER_CLASS else DEFAULT_SEQUENCES_PER_CLASS,
            output_dir=args.output or DEFAULT_SEQUENCE_OUTPUT,
            camera_index=args.camera,
        )
    else:
        run_collector(
            classes=args.classes,
            samples_per_class=args.samples,
            output_dir=args.output or DEFAULT_OUTPUT_DIR,
            camera_index=args.camera,
        )

