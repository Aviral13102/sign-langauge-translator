"""
train_lstm.py -- LSTM Training Pipeline for Dynamic ASL Gestures
=================================================================
Trains an LSTM classifier on 30-frame sequences of normalised hand
landmarks to recognise dynamic ASL gestures (hello, thank you, etc.).

Usage:
    python src/train_lstm.py
    python src/train_lstm.py --data data/sequences --epochs 100
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- Default Paths ----
DEFAULT_SEQUENCES = "data/sequences/sequences.npy"
DEFAULT_LABELS = "data/sequences/labels.npy"
MODEL_OUTPUT = "models/lstm_dynamic.pt"
ENCODER_OUTPUT = "models/lstm_label_encoder.joblib"
PLOTS_DIR = "models/plots"

# ---- Constants ----
SEQUENCE_LENGTH = 30
FEATURE_DIM = 63


def train_lstm(
    sequences_path=DEFAULT_SEQUENCES,
    labels_path=DEFAULT_LABELS,
    lstm_units=(128, 64),
    dense_units=32,
    dropout=0.3,
    max_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    test_size=0.2,
    patience=15,
):
    """
    Train a PyTorch LSTM model for dynamic gesture recognition.
    """
    print("=" * 62)
    print("  [ASL] PyTorch LSTM Dynamic Gesture Training Pipeline")
    print("=" * 62)

    # ---- Step 1: Load data ----
    print("\n[Step 1/5] Loading sequence data...")

    seq_path = Path(sequences_path)
    lbl_path = Path(labels_path)

    if not seq_path.exists():
        print(f"[ERROR] Sequences not found: {seq_path}")
        print("[INFO] Record dynamic gesture sequences first using:")
        print("       python src/collect_data.py --mode sequence")
        sys.exit(1)

    if not lbl_path.exists():
        print(f"[ERROR] Labels not found: {lbl_path}")
        sys.exit(1)

    sequences = np.load(str(seq_path))
    labels = np.load(str(lbl_path))

    print(f"[INFO] Loaded sequences: {sequences.shape}")
    print(f"[INFO] Loaded labels: {labels.shape}")
    print(f"[INFO] Unique gestures: {len(np.unique(labels))}")
    print(f"[INFO] Gestures: {sorted(np.unique(labels).tolist())}")

    # ---- Step 2: Encode labels and split ----
    print("\n[Step 2/5] Preparing data...")

    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    print(f"[INFO] Encoded {num_classes} classes: {label_encoder.classes_.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )
    
    # Split train further into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train,
    )

    print(f"[INFO] Training set:   {X_train.shape[0]} sequences")
    print(f"[INFO] Validation set: {X_val.shape[0]} sequences")
    print(f"[INFO] Test set:       {X_test.shape[0]} sequences")

    # ---- Step 3: Build PyTorch LSTM model ----
    print("\n[Step 3/5] Building LSTM model...")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

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
            
            # Take the output of the last time step
            x = x[:, -1, :]
            
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout3(x)
            x = self.fc2(x)
            return x

    model = GestureLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim1=lstm_units[0],
        hidden_dim2=lstm_units[1] if len(lstm_units) > 1 else 64,
        dense_dim=dense_units,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6)

    # ---- Step 4: Train ----
    print(f"\n[Step 4/5] Training LSTM...")
    
    start_time = time.time()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"Epoch {epoch+1}/{max_epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} (Best)")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}/{max_epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Restore best weights
    model.load_state_dict(best_model_state)

    train_time = time.time() - start_time
    print(f"\n[INFO] Training complete in {train_time:.1f}s")

    # ---- Step 5: Evaluate and save ----
    print("\n[Step 5/5] Evaluating and saving model...")

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n[RESULT] Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")

    class_names = label_encoder.classes_.tolist()
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    # Save model and encoder
    model_path = Path(MODEL_OUTPUT)
    encoder_path = Path(ENCODER_OUTPUT)
    plots_path = Path(PLOTS_DIR)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Save model weights and configuration together
    model_data = {
        "state_dict": model.state_dict(),
        "input_dim": FEATURE_DIM,
        "hidden_dim1": lstm_units[0],
        "hidden_dim2": lstm_units[1] if len(lstm_units) > 1 else 64,
        "dense_dim": dense_units,
        "num_classes": num_classes,
        "dropout": dropout
    }
    torch.save(model_data, str(model_path))
    joblib.dump(label_encoder, str(encoder_path))

    print(f"  [OK] Model saved:   {model_path}")
    print(f"  [OK] Encoder saved: {encoder_path}")

    # Save training plots
    _plot_training_history(history, plots_path)
    cm = confusion_matrix(all_targets, all_preds)
    _plot_lstm_confusion_matrix(cm, class_names, plots_path)

    # Summary
    print(f"\n{'='*62}")
    print(f"  LSTM Training Summary")
    print(f"{'='*62}")
    print(f"  Test Accuracy      : {accuracy:.4f} ({accuracy:.1%})")
    print(f"  Training Time      : {train_time:.1f}s")
    print(f"  Epochs run         : {len(history['loss'])}")
    print(f"  Model File         : {model_path}")
    print(f"{'='*62}")

    return model, label_encoder


def _plot_training_history(history, output_dir):
    """Plot LSTM training loss and accuracy curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history["loss"], label="Train Loss", color="steelblue")
    ax1.plot(history["val_loss"], label="Val Loss", color="darkorange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("LSTM Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history["accuracy"], label="Train Acc", color="steelblue")
    ax2.plot(history["val_accuracy"], label="Val Acc", color="darkorange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("LSTM Training Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    path = Path(output_dir) / "lstm_training_history.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Training history saved: {path}")


def _plot_lstm_confusion_matrix(cm, class_names, output_dir):
    """Plot LSTM confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("LSTM Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = Path(output_dir) / "lstm_confusion_matrix.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved: {path}")


# ---- CLI Entry Point -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PyTorch LSTM classifier for dynamic ASL gestures."
    )
    parser.add_argument(
        "--sequences", default=DEFAULT_SEQUENCES,
        help="Path to sequences .npy file",
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABELS,
        help="Path to labels .npy file",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)",
    )

    args = parser.parse_args()

    train_lstm(
        sequences_path=args.sequences,
        labels_path=args.labels,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
