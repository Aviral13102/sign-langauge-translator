"""
generate_synthetic_sequences.py
================================
Generates synthetic 30-frame gesture sequences for LSTM training
by simulating motion trajectories from static landmark poses.

This creates plausible dynamic gesture data without requiring
manual webcam recording of every gesture.

Strategy:
  - For each gesture class, we generate sequences by:
    1. Sampling a random start pose (e.g. rest/neutral)
    2. Sampling a random end pose from that gesture class
    3. Interpolating + adding noise to simulate natural motion
    4. Adding temporal jitter for realistic timing variation

Usage:
    python src/generate_synthetic_sequences.py
    python src/generate_synthetic_sequences.py --sequences 100
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- Defaults ----
DEFAULT_OUTPUT = "data/sequences"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 63
DEFAULT_SEQUENCES_PER_CLASS = 80

# Gesture classes for dynamic recognition
GESTURE_CLASSES = [
    "hello", "thank_you", "yes", "no", "please",
    "sorry", "help", "goodbye", "i_love_you", "stop",
]

# Map gesture names to motion patterns
# Each pattern defines how landmarks move over time
GESTURE_PATTERNS = {
    "hello": {
        "type": "wave",
        "axis": "x",
        "frequency": 3.0,
        "amplitude": 0.15,
        "fingers": "open",
    },
    "thank_you": {
        "type": "arc_forward",
        "axis": "z",
        "amplitude": 0.2,
        "fingers": "flat",
    },
    "yes": {
        "type": "nod",
        "axis": "y",
        "frequency": 2.0,
        "amplitude": 0.12,
        "fingers": "fist",
    },
    "no": {
        "type": "wave",
        "axis": "x",
        "frequency": 2.0,
        "amplitude": 0.1,
        "fingers": "index_point",
    },
    "please": {
        "type": "circle",
        "axis": "xy",
        "frequency": 1.0,
        "amplitude": 0.1,
        "fingers": "flat",
    },
    "sorry": {
        "type": "circle",
        "axis": "xy",
        "frequency": 1.5,
        "amplitude": 0.08,
        "fingers": "fist",
    },
    "help": {
        "type": "lift",
        "axis": "y",
        "amplitude": 0.2,
        "fingers": "thumbs_up",
    },
    "goodbye": {
        "type": "wave",
        "axis": "x",
        "frequency": 2.5,
        "amplitude": 0.12,
        "fingers": "open",
    },
    "i_love_you": {
        "type": "hold_rotate",
        "axis": "z",
        "amplitude": 0.05,
        "fingers": "ily",
    },
    "stop": {
        "type": "push_forward",
        "axis": "z",
        "amplitude": 0.15,
        "fingers": "open",
    },
}


def generate_base_hand(finger_config="open"):
    """
    Generate a base hand landmark vector (21 landmarks x 3 coords = 63).
    Landmarks are in normalized wrist-relative space.
    """
    rng = np.random.default_rng()

    # 21 landmarks: wrist + 4 per finger (5 fingers)
    # Rough anatomical layout in normalized space
    landmarks = np.zeros(63, dtype=np.float32)

    # Wrist at origin (already 0,0,0)

    # Finger base positions (MCP joints)
    finger_bases = {
        "thumb":  np.array([0.15, -0.05, 0.0]),
        "index":  np.array([0.12, -0.15, 0.0]),
        "middle": np.array([0.04, -0.17, 0.0]),
        "ring":   np.array([-0.04, -0.15, 0.0]),
        "pinky":  np.array([-0.12, -0.12, 0.0]),
    }

    finger_order = ["thumb", "index", "middle", "ring", "pinky"]

    for fidx, fname in enumerate(finger_order):
        base = finger_bases[fname]
        # Each finger: 4 landmarks (MCP, PIP, DIP, TIP)
        for jidx in range(4):
            lidx = 1 + fidx * 4 + jidx  # landmark index (0 = wrist)
            progress = (jidx + 1) / 4.0

            if finger_config == "open":
                # Extended fingers
                offset = base * (1.0 + progress * 0.8)
            elif finger_config == "fist":
                # Curled fingers
                curl = 0.3 * progress
                offset = base * (1.0 + progress * 0.2)
                offset[1] += curl  # curl back towards palm
            elif finger_config == "flat":
                # Flat hand
                offset = base * (1.0 + progress * 0.7)
                offset[2] -= 0.02 * progress
            elif finger_config == "index_point":
                if fname == "index":
                    offset = base * (1.0 + progress * 0.9)
                else:
                    curl = 0.25 * progress
                    offset = base * (1.0 + progress * 0.15)
                    offset[1] += curl
            elif finger_config == "thumbs_up":
                if fname == "thumb":
                    offset = base.copy()
                    offset[1] = -(0.1 + progress * 0.15)
                else:
                    curl = 0.3 * progress
                    offset = base * (1.0 + progress * 0.15)
                    offset[1] += curl
            elif finger_config == "ily":
                # I Love You: thumb, index, pinky extended; middle, ring curled
                if fname in ("thumb", "index", "pinky"):
                    offset = base * (1.0 + progress * 0.8)
                else:
                    curl = 0.3 * progress
                    offset = base * (1.0 + progress * 0.15)
                    offset[1] += curl
            else:
                offset = base * (1.0 + progress * 0.6)

            # Add small anatomical noise
            offset += rng.normal(0, 0.008, 3).astype(np.float32)

            landmarks[lidx * 3: lidx * 3 + 3] = offset

    return landmarks


def generate_motion_trajectory(pattern, num_frames=SEQUENCE_LENGTH):
    """
    Generate per-frame motion offsets based on the gesture pattern.
    Returns array of shape (num_frames, 63).
    """
    rng = np.random.default_rng()
    t = np.linspace(0, 1, num_frames)
    offsets = np.zeros((num_frames, 63), dtype=np.float32)

    motion_type = pattern["type"]
    amplitude = pattern.get("amplitude", 0.1)
    frequency = pattern.get("frequency", 1.0)

    if motion_type == "wave":
        # Side-to-side wave
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        for i in range(num_frames):
            # Apply wave to all x-coordinates
            for j in range(0, 63, 3):
                offsets[i, j] = wave[i]

    elif motion_type == "nod":
        # Up-down nodding motion
        nod = amplitude * np.sin(2 * np.pi * frequency * t)
        for i in range(num_frames):
            for j in range(1, 63, 3):  # y-coordinates
                offsets[i, j] = nod[i]

    elif motion_type == "arc_forward":
        # Forward arc (like bringing hand from chin outward)
        arc_y = -amplitude * 0.5 * t
        arc_z = amplitude * np.sin(np.pi * t)
        for i in range(num_frames):
            for j in range(1, 63, 3):
                offsets[i, j] = arc_y[i]
            for j in range(2, 63, 3):
                offsets[i, j] = arc_z[i]

    elif motion_type == "circle":
        # Circular motion on chest
        cx = amplitude * np.cos(2 * np.pi * frequency * t)
        cy = amplitude * np.sin(2 * np.pi * frequency * t)
        for i in range(num_frames):
            for j in range(0, 63, 3):
                offsets[i, j] = cx[i]
            for j in range(1, 63, 3):
                offsets[i, j] = cy[i]

    elif motion_type == "lift":
        # Upward lift
        lift = -amplitude * t  # negative y = up
        for i in range(num_frames):
            for j in range(1, 63, 3):
                offsets[i, j] = lift[i]

    elif motion_type == "push_forward":
        # Push hand forward (z-axis)
        push = amplitude * (1 - np.cos(np.pi * t)) / 2
        for i in range(num_frames):
            for j in range(2, 63, 3):
                offsets[i, j] = push[i]

    elif motion_type == "hold_rotate":
        # Slight rotation/wobble while holding a pose
        wobble = amplitude * 0.3 * np.sin(2 * np.pi * 0.5 * t)
        for i in range(num_frames):
            for j in range(2, 63, 3):
                offsets[i, j] = wobble[i]

    # Add temporal jitter
    noise = rng.normal(0, amplitude * 0.1, offsets.shape).astype(np.float32)
    offsets += noise

    return offsets


def generate_sequence(gesture_name, variation_seed=None):
    """
    Generate a single synthetic 30-frame sequence for a gesture.
    """
    if variation_seed is not None:
        np.random.seed(variation_seed)

    pattern = GESTURE_PATTERNS[gesture_name]
    finger_config = pattern.get("fingers", "open")

    # Generate base hand pose
    base = generate_base_hand(finger_config)

    # Generate motion trajectory
    motion = generate_motion_trajectory(pattern)

    # Build sequence: base + motion + per-frame noise
    rng = np.random.default_rng(variation_seed)
    sequence = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

    for i in range(SEQUENCE_LENGTH):
        frame = base.copy()

        # Apply motion
        frame += motion[i]

        # Add per-frame variation (simulates hand tremor, camera jitter)
        frame += rng.normal(0, 0.005, FEATURE_DIM).astype(np.float32)

        # Random scale variation (simulates distance changes)
        scale = rng.uniform(0.9, 1.1)
        frame *= scale

        sequence[i] = frame

    return sequence


def main(
    output_dir=DEFAULT_OUTPUT,
    sequences_per_class=DEFAULT_SEQUENCES_PER_CLASS,
    gestures=None,
):
    """Generate synthetic sequences for all gesture classes."""
    if gestures is None:
        gestures = GESTURE_CLASSES

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  [ASL] Synthetic Sequence Generator")
    print("=" * 62)
    print(f"  Gestures           : {len(gestures)}")
    print(f"  Sequences/gesture  : {sequences_per_class}")
    print(f"  Frames/sequence    : {SEQUENCE_LENGTH}")
    print(f"  Feature dim        : {FEATURE_DIM}")
    print(f"  Output             : {output_path}")
    print("=" * 62)
    print()

    all_sequences = []
    all_labels = []

    for i, gesture in enumerate(gestures):
        print(f"  [{i+1}/{len(gestures)}] Generating '{gesture}'...", end=" ")

        for j in range(sequences_per_class):
            seq = generate_sequence(gesture, variation_seed=i * 10000 + j)
            all_sequences.append(seq)
            all_labels.append(gesture)

        print(f"{sequences_per_class} sequences")

    # Save
    seq_array = np.array(all_sequences, dtype=np.float32)
    lbl_array = np.array(all_labels)

    np.save(str(output_path / "sequences.npy"), seq_array)
    np.save(str(output_path / "labels.npy"), lbl_array)

    print(f"\n{'='*62}")
    print(f"  Generation Complete")
    print(f"{'='*62}")
    print(f"  Total sequences    : {seq_array.shape[0]}")
    print(f"  Sequence shape     : {seq_array.shape}")
    print(f"  Unique gestures    : {len(set(all_labels))}")
    print(f"  Saved to           : {output_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic gesture sequences for LSTM training."
    )
    parser.add_argument(
        "--sequences", type=int, default=DEFAULT_SEQUENCES_PER_CLASS,
        help=f"Sequences per class (default: {DEFAULT_SEQUENCES_PER_CLASS})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Specific gesture classes to generate",
    )

    args = parser.parse_args()
    main(
        output_dir=args.output,
        sequences_per_class=args.sequences,
        gestures=args.classes,
    )
