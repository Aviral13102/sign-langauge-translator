"""
landmarks.py — MediaPipe Hand Landmark Detection (Tasks API)
==============================================================
Detects hands in video frames using Google's MediaPipe Hand Landmarker
(Tasks API) and extracts 21 landmark coordinates (x, y, z) per hand.
Provides drawing utilities for overlaying landmarks on frames.

Requires the hand_landmarker.task model file. The model will be
automatically downloaded on first use.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from urllib.request import urlretrieve

# ─── Constants ──────────────────────────────────────────────────────

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# MediaPipe hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),              # Palm
]


def download_model():
    """Download the hand landmarker model if not already present."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading hand landmarker model...")
    print(f"[INFO] URL: {MODEL_URL}")
    print(f"[INFO] Saving to: {MODEL_PATH}")

    try:
        urlretrieve(MODEL_URL, str(MODEL_PATH))
        print(f"[INFO] Model downloaded successfully ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        raise

    return MODEL_PATH


class HandLandmarkDetector:
    """Detects hand landmarks using MediaPipe Hand Landmarker (Tasks API)."""

    # Landmark indices for reference
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    NUM_LANDMARKS = 21
    NUM_FEATURES = 63  # 21 landmarks x 3 coordinates (x, y, z)

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ):
        """
        Initialise the MediaPipe Hand Landmarker.

        Args:
            static_image_mode: If True, treats each image independently (for dataset processing).
                               If False, uses tracking for video streams (faster).
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        # Download model if needed
        model_path = download_model()

        self._static_mode = static_image_mode
        self._frame_timestamp_ms = 0

        # Choose running mode
        if static_image_mode:
            running_mode = vision.RunningMode.IMAGE
        else:
            running_mode = vision.RunningMode.VIDEO

        # Create the hand landmarker
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Detect hands and extract landmarks from a BGR frame.

        Args:
            frame: BGR image (numpy array) from OpenCV.

        Returns:
            results: HandLandmarkerResult object.
                     results.hand_landmarks contains the landmark data.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._static_mode:
            results = self.landmarker.detect(mp_image)
        else:
            # VIDEO mode requires monotonically increasing timestamps
            self._frame_timestamp_ms += 33  # ~30 fps
            results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        return results

    def extract_landmarks(self, results, hand_index=0):
        """
        Extract landmark coordinates as a NumPy array.

        Args:
            results: HandLandmarkerResult object.
            hand_index: Which hand to extract (0 = first detected hand).

        Returns:
            numpy array of shape (21, 3) with x, y, z coordinates,
            or None if no hand detected at the given index.
        """
        if not results.hand_landmarks:
            return None

        if hand_index >= len(results.hand_landmarks):
            return None

        hand = results.hand_landmarks[hand_index]
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32,
        )
        return landmarks

    def extract_flat(self, results, hand_index=0):
        """
        Extract landmarks as a flat 63-dimensional feature vector.

        Args:
            results: HandLandmarkerResult object.
            hand_index: Which hand to extract.

        Returns:
            numpy array of shape (63,) -- flattened [x0, y0, z0, x1, y1, z1, ...],
            or None if no hand detected.
        """
        landmarks = self.extract_landmarks(results, hand_index)
        if landmarks is None:
            return None
        return landmarks.flatten()

    def draw_landmarks(self, frame, results):
        """
        Draw all detected hand landmarks and connections on the frame.

        Uses OpenCV drawing directly (no dependency on mp.solutions).

        Args:
            frame: BGR image (numpy array) to draw on.
            results: HandLandmarkerResult object.

        Returns:
            The frame with landmarks drawn (modified in-place).
        """
        if not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand in results.hand_landmarks:
            # Draw bone connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                x1 = int(hand[start_idx].x * w)
                y1 = int(hand[start_idx].y * h)
                x2 = int(hand[end_idx].x * w)
                y2 = int(hand[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)

            # Draw landmark dots (green circles)
            for lm in hand:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 180, 0), 1, cv2.LINE_AA)  # Outline

        return frame

    def draw_hand_info(self, frame, results):
        """
        Draw additional hand information (handedness label) on the frame.

        Args:
            frame: BGR image to draw on.
            results: HandLandmarkerResult object.

        Returns:
            The frame with hand info overlaid.
        """
        if not results.hand_landmarks or not results.handedness:
            return frame

        h, w, _ = frame.shape

        for hand, handedness in zip(results.hand_landmarks, results.handedness):
            # Get the label (Left/Right) and confidence
            label = handedness[0].category_name
            confidence = handedness[0].score

            # Position the text near the wrist
            wrist = hand[self.WRIST]
            text_x = int(wrist.x * w) - 30
            text_y = int(wrist.y * h) + 40

            # Ensure text stays within frame bounds
            text_x = max(10, min(text_x, w - 150))
            text_y = max(30, min(text_y, h - 10))

            hand_text = f"{label} ({confidence:.0%})"

            # Draw text with background
            (text_w, text_h), _ = cv2.getTextSize(
                hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_h - 8),
                (text_x + text_w + 5, text_y + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                hand_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return frame

    def get_num_hands(self, results):
        """Return the number of hands detected in the results."""
        if not results.hand_landmarks:
            return 0
        return len(results.hand_landmarks)

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()


# ─── Standalone Test ────────────────────────────────────────────────
if __name__ == "__main__":
    from capture import WebcamCapture

    print("[INFO] Testing MediaPipe Hand Landmark Detection (Tasks API)")
    print("[INFO] Show your hand(s) to the webcam. Press 'q' to quit.\n")

    cam = WebcamCapture()
    detector = HandLandmarkDetector()

    cam.start()
    try:
        for success, frame in cam.get_frames():
            results = detector.detect(frame)
            detector.draw_landmarks(frame, results)
            detector.draw_hand_info(frame, results)
            cam.draw_fps(frame)

            # Show detection status
            num_hands = detector.get_num_hands(results)
            status = f"Hands: {num_hands}"
            cv2.putText(
                frame, status, (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )

            cv2.imshow(cam.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cam.stop()
