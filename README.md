# 🤟 Real-time Sign Language Translator

A computer vision application that translates American Sign Language (ASL) gestures into text and speech in real time using your webcam.

## Features

- **Static Letters (A-Z)**: Recognises 28 classes (alphabet + space + delete) with **97.3% accuracy** using a lightweight MLP classifier.
- **Dynamic Gestures**: Detects 10 common moving signs (hello, thank you, please, etc.) with **100% test accuracy** using a PyTorch LSTM.
- **Word Builder**: Assembles fingerspelled letters into complete words with majority-vote smoothing and hold-to-confirm logic to eliminate flickering.
- **Text-to-Speech (TTS)**: Speaks translated words and gestures aloud automatically.
- **Web & Desktop UIs**: Run as a standalone Python desktop app with a custom HUD, or as a modern Flask web application.
- **Position Invariant**: Uses wrist-relative normalization so gestures are recognized accurately regardless of where your hand is on the screen.

## Architecture

```mermaid
graph TD
    A[Webcam] -->|RGB Frame| B(OpenCV)
    B -->|Pre-processed Frame| C(MediaPipe Hands)
    C -->|21 Joint Landmarks| D[Normalization]
    D -->|63-dim Vector| E{Gesture Type}
    
    E -->|Static (Letter)| F(MLP Classifier)
    E -->|Dynamic (Sequence)| G(LSTM Classifier)
    
    F -->|Class + Confidence| H[Word Builder]
    G -->|Gesture + Confidence| I[Gesture Banner]
    
    H -->|Majority Vote| J[Hold-to-Confirm]
    J -->|Confirmed Letter| K[Word Buffer]
    
    K -->|Display| L[HUD / Web UI]
    K -->|Text-to-Speech| M[pyttsx3 / Web Speech API]
    I -->|Display & Speak| M
```

## Model Performance

| Model | Task | Architecture | Test Accuracy |
|-------|------|--------------|---------------|
| **Static MLP** | Letters (28 classes) | Input(63) → Dense(256) → Dense(128) → Output(28) | **97.3%** |
| **Dynamic LSTM** | Gestures (10 classes) | Input(30,63) → LSTM(128) → LSTM(64) → Dense(32) → Output(10) | **100.0%** |

*Note: The MLP model includes robustness testing proving >95% accuracy against 10% scale variations and 5-degree rotations.*

## Installation

**Prerequisites**: Python 3.10+ and a webcam.

```bash
# Clone the repository
git clone https://github.com/yourusername/sign-language-translator.git
cd sign-language-translator

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Modern Web App (Recommended)
Run the Flask server and access the translator through your browser.

```bash
python web/app_web.py
```
Then open `http://localhost:5000` in your web browser.

### Option 2: Desktop App (OpenCV HUD)
Run the standalone Python application with real-time on-screen overlays.

```bash
python app.py
```

**Desktop Keyboard Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Confirm current word and speak it |
| `BACKSPACE`| Delete the last letter |
| `c` | Clear the current word buffer |
| `r` | Reset entire translation history |
| `t` | Toggle Text-to-Speech (TTS) ON/OFF |
| `q` | Quit the application |

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **MediaPipe** | High-speed 21-point hand landmark detection |
| **OpenCV** | Webcam interfacing and desktop HUD rendering |
| **scikit-learn**| MLP classifier for static alphabet recognition |
| **PyTorch** | LSTM sequence model for dynamic gesture recognition |
| **Flask** | Web server backend and REST API for predictions |
| **pyttsx3** | Offline Text-to-Speech engine for desktop app |

## Project Structure

```
sign-language-translator/
├── app.py                      # Main desktop application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── landmarks/              # Processed 63-dim vectors (.npy)
│   └── sequences/              # 30-frame gesture sequences
├── docs/
│   └── architecture.md         # Mermaid architecture diagrams
├── models/
│   ├── mlp_static.joblib       # Trained scikit-learn MLP model
│   ├── lstm_dynamic.pt         # Trained PyTorch LSTM model
│   ├── label_encoder.joblib    # Alphabet class mapping
│   ├── scaler.joblib           # Feature scaling weights
│   └── plots/                  # Training curves & confusion matrices
├── src/
│   ├── augment.py              # Data augmentation (noise, scale, rotate)
│   ├── capture.py              # Base webcam capture logic
│   ├── collect_data.py         # Custom dataset recording tool
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── evaluate.py             # Model evaluation scripts
│   ├── extract_landmarks.py    # Batch MediaPipe extraction
│   ├── generate_synthetic_sequences.py # Generates training sequences
│   ├── landmarks.py            # MediaPipe wrapper class
│   ├── normalize.py            # Wrist-relative coordinate scaling
│   ├── robustness_test.py      # Automated stress testing script
│   ├── sequence_detector.py    # Runtime LSTM sliding window inference
│   ├── train.py                # MLP training pipeline
│   ├── train_lstm.py           # PyTorch LSTM training pipeline
│   ├── tts.py                  # Background pyttsx3 threading
│   └── word_builder.py         # Majority-vote & debounce logic
└── web/                        
    ├── app_web.py              # Flask server entry point
    ├── static/
    │   ├── css/style.css       # Web UI styling
    │   └── js/translator.js    # WebRTC & client-side inference logic
    └── templates/
        ├── index.html          # Landing page
        └── translate.html      # Web translation interface
```

## Limitations & Future Work

- **Vocabulary Size**: The dynamic model currently recognizes 10 gestures. Expanding this requires significantly more diverse training sequences.
- **Lighting Sensitivity**: Like all computer vision approaches, extreme low-light conditions will degrade MediaPipe's ability to detect hand landmarks.
- **Two-Handed Signs**: This project currently focuses on single-hand ASL fingerspelling and signs. Full ASL requires tracking both hands simultaneously.

## Credits

Developed by Aviral Singh (23BBS0156) for CBS3006 Machine Learning.
