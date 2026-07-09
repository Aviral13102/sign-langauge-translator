"""
app.py -- Sign Language Translator: Live Demo
===============================================
Main entry point combining webcam capture, MediaPipe hand landmark
detection, ASL letter prediction, word assembly, and text-to-speech.

Phase 1: Landmarks only (green dots + FPS)
Phase 2: + Predicted letter with confidence bar
Phase 3: + Word builder, majority-vote smoothing, hold-to-confirm, TTS

Usage:
    python app.py                  # Run with default camera
    python app.py --camera 1       # Use camera at index 1
    python app.py --no-model       # Force Phase 1 mode (landmarks only)
    python app.py --no-tts         # Disable text-to-speech

Controls:
    q         -- Quit
    c         -- Clear current word
    r         -- Reset all history
    t         -- Toggle TTS on/off
    SPACE     -- Confirm word + speak (TTS)
    BACKSPACE -- Delete last letter
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from src.capture import WebcamCapture
from src.landmarks import HandLandmarkDetector
from src.normalize import normalize_single_frame
from src.word_builder import WordBuilder
from src.tts import TTSEngine
from src.sequence_detector import SequenceDetector

# ---- Model Paths ----
MODEL_PATH = Path("models/mlp_static.joblib")
ENCODER_PATH = Path("models/label_encoder.joblib")
SCALER_PATH = Path("models/scaler.joblib")


def load_model():
    """
    Attempt to load the trained MLP model and preprocessing artifacts.

    Returns:
        Tuple of (model, label_encoder, scaler) or (None, None, None)
        if the model is not found.
    """
    if not MODEL_PATH.exists():
        return None, None, None

    try:
        import joblib
        model = joblib.load(str(MODEL_PATH))
        label_encoder = joblib.load(str(ENCODER_PATH))
        scaler = joblib.load(str(SCALER_PATH))
        print(f"[INFO] Model loaded: {MODEL_PATH}")
        print(f"[INFO] Classes: {label_encoder.classes_.tolist()}")
        return model, label_encoder, scaler
    except Exception as e:
        print(f"[WARNING] Failed to load model: {e}")
        return None, None, None


# ---- HUD Drawing Functions -------------------------------------------------

def draw_prediction_box(frame, letter, confidence, threshold=0.85):
    """Draw the predicted letter and confidence in the top-right corner."""
    h, w, _ = frame.shape

    if confidence < threshold or letter is None:
        # Below threshold - show faded "?" indicator
        cv2.rectangle(frame, (w - 160, 10), (w - 10, 110), (0, 0, 0), -1)
        cv2.putText(
            frame, "?", (w - 110, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (80, 80, 80), 4, cv2.LINE_AA,
        )
        return

    # Background
    cv2.rectangle(frame, (w - 160, 10), (w - 10, 130), (0, 0, 0), -1)

    # Large predicted letter
    color = (0, 255, 0) if confidence >= 0.95 else (0, 200, 255) if confidence >= 0.85 else (0, 100, 255)
    cv2.putText(
        frame, letter, (w - 120, 85),
        cv2.FONT_HERSHEY_SIMPLEX, 2.8, color, 4, cv2.LINE_AA,
    )

    # Confidence percentage
    cv2.putText(
        frame, f"{confidence:.0%}", (w - 150, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )


def draw_hold_progress_bar(frame, progress, candidate_letter):
    """Draw the hold-to-confirm progress bar at the bottom of the frame."""
    h, w, _ = frame.shape

    bar_x = 10
    bar_y = h - 75
    bar_w = w - 20
    bar_h = 20

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)

    if progress > 0:
        # Filled portion
        fill_w = int(bar_w * progress)
        # Colour transitions: grey → yellow → green
        if progress < 0.5:
            bar_color = (0, 180, 255)  # orange
        elif progress < 1.0:
            bar_color = (0, 230, 255)  # yellow
        else:
            bar_color = (0, 255, 0)  # green = confirmed

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Label
        label = f"Hold '{candidate_letter}': {progress:.0%}"
        cv2.putText(
            frame, label, (bar_x + 5, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )


def draw_word_panel(frame, state, tts_enabled):
    """
    Draw the word assembly panel at the bottom of the frame.

    Shows: current word, history, and controls help.
    """
    h, w, _ = frame.shape

    # ---- Word panel background ----
    panel_y = h - 50
    cv2.rectangle(frame, (0, panel_y), (w, h), (20, 20, 20), -1)

    # ---- Current word (large, centered) ----
    current_word = state["current_word"]
    if current_word:
        # Blinking cursor effect
        import time
        cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
        word_display = current_word + cursor
    else:
        word_display = "..."

    cv2.putText(
        frame, word_display, (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # ---- History (scrolling text on the right) ----
    full_text = state["full_text"]
    if full_text and full_text != current_word:
        # Show history before current word
        history_text = " ".join(state["history"])
        if len(history_text) > 40:
            history_text = "..." + history_text[-37:]

        cv2.putText(
            frame, history_text, (15, panel_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
        )

    # ---- TTS indicator ----
    tts_text = "TTS: ON" if tts_enabled else "TTS: OFF"
    tts_color = (0, 200, 0) if tts_enabled else (100, 100, 100)
    cv2.putText(
        frame, tts_text, (w - 100, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_color, 1, cv2.LINE_AA,
    )


def draw_controls_help(frame):
    """Draw keyboard controls help in the top-left area."""
    h, w, _ = frame.shape

    controls = [
        "q:Quit  c:Clear  r:Reset",
        "t:TTS  SPACE:Confirm  BS:Delete",
    ]

    y_start = 130
    for i, line in enumerate(controls):
        cv2.putText(
            frame, line, (15, y_start + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA,
        )


def draw_gesture_banner(frame, gesture, confidence, elapsed, display_duration=2.0):
    """Draw a large banner when a dynamic gesture is detected."""
    h, w, _ = frame.shape
    if elapsed > display_duration:
        return

    # Fade out effect
    alpha = max(0.0, 1.0 - (elapsed / display_duration))
    green_val = int(255 * alpha)

    banner_h = 60
    banner_y = h // 2 - banner_h // 2 - 60

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7 * alpha, frame, 1.0 - 0.7 * alpha, 0, frame)

    # Gesture text
    gesture_display = gesture.replace("_", " ").upper()
    text = f"[GESTURE] {gesture_display} ({confidence:.0%})"
    cv2.putText(
        frame, text, (w // 2 - 250, banner_y + 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, green_val, 0), 2, cv2.LINE_AA,
    )


def draw_confidence_bar(frame, confidence, threshold):
    """Draw the confidence bar above the word panel."""
    h, w, _ = frame.shape

    bar_x = 10
    bar_y = h - 105
    bar_w = 250
    bar_h = 20

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)

    # Filled portion
    fill_w = int(bar_w * confidence)
    bar_color = (0, 255, 0) if confidence >= 0.95 else (0, 200, 255) if confidence >= 0.85 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # Threshold marker
    thresh_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 255), 2)

    # Label
    cv2.putText(
        frame, f"Conf: {confidence:.0%}", (bar_x + bar_w + 10, bar_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )


# ---- Main Demo Loop -------------------------------------------------------

def run_live_demo(
    camera_index=0,
    detection_confidence=0.7,
    tracking_confidence=0.5,
    prediction_threshold=0.85,
    use_model=True,
    use_tts=True,
):
    """
    Run the live demo with gesture prediction, word assembly, and TTS.

    Args:
        camera_index: Camera device index.
        detection_confidence: MediaPipe hand detection confidence.
        tracking_confidence: MediaPipe hand tracking confidence.
        prediction_threshold: Minimum confidence to accept predictions.
        use_model: Whether to attempt loading the trained model.
        use_tts: Whether to enable text-to-speech.
    """
    window_name = "ASL Sign Language Translator"

    print("=" * 62)
    print("  [ASL]  Real-time Sign Language Translator  --  Phase 3")
    print("=" * 62)
    print(f"  Camera index          : {camera_index}")
    print(f"  Detection confidence  : {detection_confidence}")
    print(f"  Prediction threshold  : {prediction_threshold}")
    print(f"  TTS enabled           : {use_tts}")
    print()
    print("  Controls:")
    print("    q         : Quit")
    print("    c         : Clear current word")
    print("    r         : Reset all history")
    print("    t         : Toggle TTS on/off")
    print("    SPACE     : Confirm word + speak")
    print("    BACKSPACE : Delete last letter")
    print("=" * 62)
    print()

    # ---- Load model ----
    model, label_encoder, scaler = (None, None, None)
    if use_model:
        model, label_encoder, scaler = load_model()

    has_model = model is not None
    phase = "Phase 3: Word Builder" if has_model else "Phase 1: Landmark Detection"
    print(f"[INFO] Running in: {phase}")

    # ---- Initialise components ----
    cam = WebcamCapture(camera_index=camera_index, window_name=window_name)
    detector = HandLandmarkDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    word_builder = WordBuilder(
        buffer_size=12,
        confirm_seconds=0.5,
        confidence_threshold=prediction_threshold,
    )

    tts = TTSEngine(enabled=use_tts)

    # LSTM dynamic gesture detector (optional)
    seq_detector = SequenceDetector()
    has_lstm = seq_detector.available
    if has_lstm:
        phase = "Phase 3: Static + Dynamic"

    cam.start()
    print("[INFO] Webcam started. Show your hand(s) to the camera!\n")

    # Track current prediction for HUD
    current_letter = None
    current_confidence = 0.0
    last_gesture = None
    last_gesture_conf = 0.0
    last_gesture_time = 0.0

    try:
        for success, frame in cam.get_frames():
            # ---- Detect hands and landmarks ----
            results = detector.detect(frame)

            # ---- Draw landmarks (green dots + bones) ----
            detector.draw_landmarks(frame, results)

            # ---- Live prediction + word assembly (Phase 3) ----
            num_hands = detector.get_num_hands(results)

            if has_model and num_hands > 0:
                landmarks_21x3 = detector.extract_landmarks(results, hand_index=0)
                if landmarks_21x3 is not None:
                    # Normalize -> Scale -> Predict (static MLP)
                    feature_vec = normalize_single_frame(landmarks_21x3)
                    feature_vec_scaled = scaler.transform(feature_vec.reshape(1, -1))

                    proba = model.predict_proba(feature_vec_scaled)[0]
                    pred_idx = np.argmax(proba)
                    current_confidence = proba[pred_idx]
                    current_letter = label_encoder.inverse_transform([pred_idx])[0]

                    # Feed to word builder
                    confirmed = word_builder.add_prediction(current_letter, current_confidence)

                    if confirmed:
                        print(f"  [CONFIRMED] Letter: {confirmed}  |  Word: '{word_builder.get_current_word()}'")

                    # Feed to LSTM sequence detector (dynamic gestures)
                    if has_lstm:
                        gesture_result = seq_detector.add_frame(feature_vec)
                        if gesture_result:
                            import time as _time
                            last_gesture = gesture_result["gesture"]
                            last_gesture_conf = gesture_result["confidence"]
                            last_gesture_time = _time.time()
                            print(f"  [GESTURE] Detected: '{last_gesture}' ({last_gesture_conf:.0%})")
                            tts.speak(last_gesture.replace("_", " "))
            else:
                # No hand detected -- decay the vote buffer
                word_builder.add_no_hand()
                if has_lstm:
                    seq_detector.add_no_hand()
                current_letter = None
                current_confidence = 0.0

            # ---- Get word builder state for HUD ----
            wb_state = word_builder.get_state()

            # ---- Draw HUD ----

            # FPS
            cam.draw_fps(frame)

            # Hand count
            status_color = (0, 255, 0) if num_hands > 0 else (100, 100, 100)
            cv2.rectangle(frame, (10, 55), (280, 85), (0, 0, 0), -1)
            cv2.putText(
                frame, f"Hands: {num_hands}", (15, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA,
            )

            # Phase indicator
            cv2.rectangle(frame, (10, 90), (360, 120), (0, 0, 0), -1)
            cv2.putText(
                frame, phase, (15, 113),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
            )

            if has_model:
                # Prediction box (top-right)
                draw_prediction_box(frame, current_letter, current_confidence, prediction_threshold)

                # Confidence bar
                if current_confidence > 0:
                    draw_confidence_bar(frame, current_confidence, prediction_threshold)

                # Hold-to-confirm progress bar
                hold_progress = wb_state["hold_progress"]
                if hold_progress > 0 and wb_state["current_letter"]:
                    draw_hold_progress_bar(frame, hold_progress, wb_state["current_letter"])

                # Word panel (bottom)
                draw_word_panel(frame, wb_state, tts.enabled)

                # Controls help
                draw_controls_help(frame)

                # Dynamic gesture banner
                if last_gesture:
                    import time as _time
                    elapsed = _time.time() - last_gesture_time
                    draw_gesture_banner(frame, last_gesture, last_gesture_conf, elapsed)
                    if elapsed > 2.0:
                        last_gesture = None

            # ---- Display the frame ----
            cv2.imshow(window_name, frame)

            # ---- Handle keypresses ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n[INFO] Quit signal received. Shutting down...")
                break

            elif key == ord("c"):
                word_builder.clear_word()
                print("[INFO] Current word cleared.")

            elif key == ord("r"):
                word_builder.clear_history()
                print("[INFO] All history reset.")

            elif key == ord("t"):
                state = tts.toggle()
                print(f"[INFO] TTS {'enabled' if state else 'disabled'}.")

            elif key == ord(" "):
                # SPACE — confirm word + TTS
                word = word_builder.confirm_word()
                if word:
                    print(f"  [WORD] Confirmed: '{word}'")
                    tts.speak(word)

            elif key == 8:  # BACKSPACE
                word_builder.delete_last()
                print("[INFO] Deleted last letter.")

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt. Shutting down...")
    finally:
        tts.shutdown()
        detector.close()
        cam.stop()
        print("[INFO] Cleanup complete. Goodbye!")


# ---- CLI Entry Point -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASL Sign Language Translator - Live Demo (Phase 3)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7,
        help="Hand detection confidence threshold (default: 0.7)",
    )
    parser.add_argument(
        "--tracking", type=float, default=0.5,
        help="Hand tracking confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Prediction confidence threshold (default: 0.85)",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Force Phase 1 mode (landmarks only, no prediction)",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable text-to-speech output",
    )

    args = parser.parse_args()

    run_live_demo(
        camera_index=args.camera,
        detection_confidence=args.confidence,
        tracking_confidence=args.tracking,
        prediction_threshold=args.threshold,
        use_model=not args.no_model,
        use_tts=not args.no_tts,
    )
