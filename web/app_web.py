"""
app_web.py -- Flask Web App for Sign Language Translator
=========================================================
Browser-based sign language translator using WebRTC for webcam
access and a REST API for real-time predictions.

Usage:
    python web/app_web.py
    # Then open http://localhost:5000 in your browser

Endpoints:
    GET  /              Landing page
    GET  /translate      Main translation interface
    POST /api/predict    Predict ASL letter from base64 frame
    GET  /api/status     Health check + model info
"""

import sys
import base64
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

# ---- Flask App ----
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ---- Global Model State ----
_model = None
_label_encoder = None
_scaler = None
_detector = None
_model_loaded = False
_classes = []


def load_models():
    """Load MLP model and MediaPipe detector on first request."""
    global _model, _label_encoder, _scaler, _detector, _model_loaded, _classes

    if _model_loaded:
        return True

    try:
        import joblib
        from src.landmarks import HandLandmarkDetector

        model_path = PROJECT_ROOT / "models" / "mlp_static.joblib"
        encoder_path = PROJECT_ROOT / "models" / "label_encoder.joblib"
        scaler_path = PROJECT_ROOT / "models" / "scaler.joblib"

        if not model_path.exists():
            print("[ERROR] Model not found. Train the model first.")
            return False

        _model = joblib.load(str(model_path))
        _label_encoder = joblib.load(str(encoder_path))
        _scaler = joblib.load(str(scaler_path))
        _classes = _label_encoder.classes_.tolist()

        _detector = HandLandmarkDetector(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

        _model_loaded = True
        print(f"[OK] Models loaded. Classes: {len(_classes)}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return False


# ---- Routes ----

@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/translate")
def translate():
    """Main translation page."""
    return render_template("translate.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept a base64-encoded JPEG frame, run prediction.

    Request JSON:
        {"frame": "data:image/jpeg;base64,..."}

    Response JSON:
        {"letter": "A", "confidence": 0.95, "landmarks": [...], "hand_detected": true}
    """
    if not load_models():
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data"}), 400

        # Decode base64 frame
        frame_data = data["frame"]
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Detect hand landmarks
        from src.normalize import normalize_single_frame

        results = _detector.detect(frame)
        num_hands = _detector.get_num_hands(results)

        if num_hands == 0:
            return jsonify({
                "hand_detected": False,
                "letter": None,
                "confidence": 0.0,
                "landmarks": None,
            })

        # Extract and normalize landmarks
        landmarks_21x3 = _detector.extract_landmarks(results, hand_index=0)
        if landmarks_21x3 is None:
            return jsonify({
                "hand_detected": False,
                "letter": None,
                "confidence": 0.0,
                "landmarks": None,
            })

        feature_vec = normalize_single_frame(landmarks_21x3)
        feature_vec_scaled = _scaler.transform(feature_vec.reshape(1, -1))

        # Predict
        proba = _model.predict_proba(feature_vec_scaled)[0]
        pred_idx = np.argmax(proba)
        confidence = float(proba[pred_idx])
        letter = _label_encoder.inverse_transform([pred_idx])[0]

        # Return landmarks for client-side drawing
        landmarks_list = landmarks_21x3.tolist()

        return jsonify({
            "hand_detected": True,
            "letter": letter,
            "confidence": confidence,
            "landmarks": landmarks_list,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def status():
    """Health check and model info."""
    load_models()
    return jsonify({
        "status": "ok" if _model_loaded else "no_model",
        "model_loaded": _model_loaded,
        "classes": _classes,
        "num_classes": len(_classes),
        "timestamp": time.time(),
    })


# ---- Entry Point ----
if __name__ == "__main__":
    print("=" * 62)
    print("  [ASL] Sign Language Translator -- Web App")
    print("=" * 62)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 62)

    # Pre-load models
    load_models()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
