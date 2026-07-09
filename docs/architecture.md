# Architecture Diagrams

## 1. End-to-End Pipeline

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
    
    K -->|Display| L[HUD overlay]
    K -->|Text-to-Speech| M[pyttsx3 / Web Speech]
    I -->|Display & Speak| M
```

## 2. MLP Architecture (Static Letters)

```mermaid
graph LR
    A[Input: 63-dim Vector] --> B[Dense: 256 units]
    B --> C[ReLU]
    C --> D[Dense: 128 units]
    D --> E[ReLU]
    E --> F[Dense: 28 units]
    F --> G[Softmax Output]
```

## 3. LSTM Architecture (Dynamic Gestures)

```mermaid
graph LR
    A[Input: 30 frames x 63 features] --> B[LSTM: 128 units]
    B --> C[Dropout: 0.3]
    C --> D[LSTM: 64 units]
    D --> E[Dropout: 0.3]
    E --> F[Linear: 32 units]
    F --> G[ReLU]
    G --> H[Dropout: 0.15]
    H --> I[Linear: 10 units]
    I --> J[Softmax Output]
```

## 4. Data Flow (Training vs Inference)

```mermaid
graph TD
    subgraph Training
    A1[Raw Images / Sequences] --> B1[Landmark Extraction]
    B1 --> C1[Normalization]
    C1 --> D1[Augmentation]
    D1 --> E1[Train Model]
    E1 --> F1[(Saved Weights .joblib / .pt)]
    end
    
    subgraph Inference
    F1 -.-> F2[Load Model]
    A2[Live Webcam Frame] --> B2[MediaPipe Detect]
    B2 --> C2[Normalization]
    C2 --> D2[Predict]
    D2 --> E2[Smoothing Buffer]
    end
```
