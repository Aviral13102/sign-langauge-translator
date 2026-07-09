"""
sequence_detector.py -- Runtime LSTM Sequence Classifier
==========================================================
Maintains a sliding window of landmark frames and runs LSTM
inference to classify dynamic ASL gestures (hello, thank you, etc.).

Usage:
    from src.sequence_detector import SequenceDetector

    seq = SequenceDetector()
    result = seq.add_frame(landmarks_63)
    if result:
        print(f"Gesture: {result['gesture']} ({result['confidence']:.0%})")
"""

import time
import numpy as np
from pathlib import Path
from collections import deque


# ---- Default Paths ----
LSTM_MODEL_PATH = Path("models/lstm_dynamic.pt")
LSTM_ENCODER_PATH = Path("models/lstm_label_encoder.joblib")

# ---- Constants ----
SEQUENCE_LENGTH = 30    # 30 frames = 1 second at 30fps
FEATURE_DIM = 63        # 21 landmarks * 3 coords
MOTION_THRESHOLD = 0.08 # Minimum motion energy to trigger LSTM
COOLDOWN_SECONDS = 3.0  # Minimum seconds between gesture detections


class SequenceDetector:
    """
    Runtime dynamic gesture detector using a PyTorch LSTM model.

    Maintains a sliding window of normalised landmark frames. When
    sufficient motion is detected (i.e. hand is moving, not just
    holding a static pose), the LSTM classifier is invoked.
    """

    def __init__(
        self,
        model_path=LSTM_MODEL_PATH,
        encoder_path=LSTM_ENCODER_PATH,
        sequence_length=SEQUENCE_LENGTH,
        confidence_threshold=0.90,
        motion_threshold=MOTION_THRESHOLD,
        cooldown_seconds=COOLDOWN_SECONDS,
    ):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.motion_threshold = motion_threshold
        self.cooldown_seconds = cooldown_seconds

        self._buffer = deque(maxlen=sequence_length)
        self._model = None
        self._encoder = None
        self._available = False
        self._device = None
        self._softmax = None
        self._last_detection_time = 0.0

        # Try to load the model
        self._load_model(model_path, encoder_path)

    def _load_model(self, model_path, encoder_path):
        """Attempt to load the PyTorch LSTM model and encoder."""
        model_path = Path(model_path)
        encoder_path = Path(encoder_path)

        if not model_path.exists():
            print(f"[INFO] LSTM model not found: {model_path}")
            print("[INFO] Dynamic gesture detection disabled.")
            return

        try:
            import torch
            import torch.nn as nn
            import joblib
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._softmax = nn.Softmax(dim=1)

            # Define the architecture identically to train_lstm.py
            class GestureLSTM(nn.Module):
                def __init__(self, input_dim, hidden_dim1, hidden_dim2, dense_dim, num_classes, dropout):
                    super(GestureLSTM, self).__init__()
                    self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
                    self.dropout1 = nn.Dropout(dropout)
                    self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
                    self.dropout2 = nn.Dropout(dropout)
                    self.fc1 = nn.Linear(hidden_dim2, dense_dim)
                    self.relu = nn.ReLU()
                    self.dropout3 = nn.Dropout(dropout * 0.5)
                    self.fc2 = nn.Linear(dense_dim, num_classes)

                def forward(self, x):
                    x, _ = self.lstm1(x)
                    x = self.dropout1(x)
                    x, _ = self.lstm2(x)
                    x = self.dropout2(x)
                    
                    x = x[:, -1, :]
                    
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.dropout3(x)
                    x = self.fc2(x)
                    return x

            # Load the saved state dict and configuration
            model_data = torch.load(str(model_path), map_location=self._device)
            
            self._model = GestureLSTM(
                input_dim=model_data.get("input_dim", FEATURE_DIM),
                hidden_dim1=model_data.get("hidden_dim1", 128),
                hidden_dim2=model_data.get("hidden_dim2", 64),
                dense_dim=model_data.get("dense_dim", 32),
                num_classes=model_data.get("num_classes"),
                dropout=model_data.get("dropout", 0.3)
            ).to(self._device)
            
            self._model.load_state_dict(model_data["state_dict"])
            self._model.eval()
            
            self._encoder = joblib.load(str(encoder_path))
            self._available = True
            
            print(f"[INFO] PyTorch LSTM model loaded: {model_path}")
            print(f"[INFO] Dynamic gestures: {self._encoder.classes_.tolist()}")
            
        except ImportError:
            print("[WARNING] PyTorch not installed. Dynamic gestures disabled.")
        except Exception as e:
            print(f"[WARNING] Failed to load PyTorch LSTM model: {e}")

    @property
    def available(self):
        """Whether the LSTM model is loaded and ready."""
        return self._available

    def add_frame(self, landmarks_63):
        """
        Add a single frame's normalised landmarks to the buffer.
        """
        if not self._available:
            return None

        self._buffer.append(landmarks_63.copy())

        # Need a full window to classify
        if len(self._buffer) < self.sequence_length:
            return None

        # Cooldown: skip if we recently detected a gesture
        if (time.time() - self._last_detection_time) < self.cooldown_seconds:
            return None

        # Check motion energy
        if not self._has_motion():
            return None

        # Run inference
        return self._predict()

    def add_no_hand(self):
        """Clear the buffer when no hand is detected."""
        self._buffer.clear()

    def _has_motion(self):
        """
        Check if the hand is moving (not just a static pose).
        """
        frames = list(self._buffer)
        if len(frames) < 2:
            return False

        displacements = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i] - frames[i - 1])
            displacements.append(np.mean(diff))

        motion_energy = np.mean(displacements)
        return motion_energy > self.motion_threshold

    def _predict(self):
        """Run LSTM inference on the current buffer."""
        import torch
        
        sequence = np.array(list(self._buffer), dtype=np.float32)
        sequence = sequence.reshape(1, self.sequence_length, FEATURE_DIM)
        
        # Convert to tensor
        tensor_seq = torch.tensor(sequence).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(tensor_seq)
            probs = self._softmax(outputs)[0].cpu().numpy()
            
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]

        if confidence < self.confidence_threshold:
            return None

        gesture = self._encoder.inverse_transform([pred_idx])[0]

        # Clear buffer and record time to enforce cooldown
        self._buffer.clear()
        self._last_detection_time = time.time()

        return {
            "gesture": gesture,
            "confidence": float(confidence),
        }

    def get_motion_energy(self):
        """Get current motion energy for debugging/display."""
        if len(self._buffer) < 2:
            return 0.0

        frames = list(self._buffer)
        displacements = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i] - frames[i - 1])
            displacements.append(np.mean(diff))

        return float(np.mean(displacements))

    def get_buffer_fill(self):
        """Get buffer fill percentage (0.0 to 1.0)."""
        return len(self._buffer) / self.sequence_length
