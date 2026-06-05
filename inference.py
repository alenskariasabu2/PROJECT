"""
inference.py
------------
Real-time scent classification from BME688 over USB serial.
Arduino must be flashed with TRAINING_MODE = true.

Required files in the same folder:
    enose_rf_model.pkl / enose_knn_model.pkl / enose_svm_model.pkl
    enose_mlp_model.pkl / enose_ensemble_model.pkl
    enose_lstm_model.pt
    enose_scaler.pkl
    enose_le.pkl
    feature_columns.pkl

Usage:
    python inference.py --port COM8                   (RF by default)
    python inference.py --port COM8 --model knn
    python inference.py --port COM8 --model svm
    python inference.py --port COM8 --model mlp
    python inference.py --port COM8 --model ensemble
    python inference.py --port COM8 --model lstm
"""

import serial
import json
import numpy as np
import pickle
import joblib
import argparse
from datetime import datetime
from collections import deque


# ── Model loading ──────────────────────────────────────────────────────────────

def load_sklearn_model(model_type):
    model_file = f'enose_{model_type}_model.pkl'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open('enose_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    features = joblib.load('feature_columns.pkl')
    print(f"Model loaded: {model_file}")
    print(f"Classes:  {list(model.classes_)}")
    print(f"Features: {len(features)} (8 gas + 8 gas_index + 8 log)\n")
    return model, scaler, features


def load_lstm_model():
    import torch
    import torch.nn as nn

    checkpoint = torch.load('enose_lstm_model.pt', map_location='cpu')

    class ENoseLSTM(nn.Module):
        def __init__(self, n_features, hidden_size, n_layers, n_classes, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    model = ENoseLSTM(
        n_features=checkpoint['n_features'],
        hidden_size=checkpoint['hidden_size'],
        n_layers=checkpoint['n_layers'],
        n_classes=checkpoint['n_classes'],
        dropout=checkpoint['dropout']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    with open('enose_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('enose_le.pkl', 'rb') as f:
        le = pickle.load(f)

    features = joblib.load('feature_columns.pkl')
    seq_len  = checkpoint['seq_len']
    labels   = checkpoint['labels']

    print(f"LSTM model loaded.")
    print(f"Sequence length: {seq_len} steps")
    print(f"Features: {len(features)} (8 gas + 8 gas_index + 8 log)")
    print(f"Classes: {labels}\n")

    return model, scaler, le, features, seq_len, labels


# ── Feature extraction ─────────────────────────────────────────────────────────

def parse_payload(payload, features):
    """
    Build a 24-feature vector from the Arduino JSON payload:
      - sensor_0_gas to sensor_7_gas         (8 raw gas resistance values)
      - sensor_0_gas_index to sensor_7_gas_index (8 heater step indices, 0-9)
      - sensor_0_gas_log to sensor_7_gas_log (8 log-transformed gas resistance)

    Gas index is included because 4 different heater profiles run across the
    8 sensors simultaneously, so each sensor is at a different heater step.
    Without gas index the model cannot contextualise the temperature at which
    each resistance reading was produced.
    """
    gas  = payload.get('g',  [])
    gi   = payload.get('gi', [])
    temp = payload.get('t',  [])
    hum  = payload.get('h',  [])

    row = {}
    for i in range(8):
        raw_gas = gas[i] if i < len(gas) and gas[i] is not None else 0
        gas_idx = gi[i]  if i < len(gi)  and gi[i]  is not None else 0

        row[f'sensor_{i}_gas']       = raw_gas
        row[f'sensor_{i}_gas_index'] = gas_idx
        row[f'sensor_{i}_gas_log']   = np.log1p(raw_gas)

        # Kept in row dict but only used if feature_cols includes them
        row[f'sensor_{i}_temp']      = temp[i] if i < len(temp) and temp[i] is not None else 0
        row[f'sensor_{i}_humidity']  = hum[i]  if i < len(hum)  and hum[i]  is not None else 0

    return np.array([[row.get(f, 0) for f in features]])


# ── Inference loops ────────────────────────────────────────────────────────────

EMOJIS = {
    'air':     '💨',
    'alcohol': '🥃',
    'coffee':  '☕',
    'vinegar': '🧪',
    'wine':    '🍷'
}


def run_sklearn(port, baud, model, scaler, features):
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected. Running inference — press Ctrl+C to stop.\n")

    history       = deque(maxlen=5)
    current       = "unknown"
    exposure_time = None
    detected      = False

    input("Place sensor near the smell, then press ENTER to start timing...\n")
    exposure_time = datetime.now()
    print(f"Timer started at {exposure_time.strftime('%H:%M:%S')}\n")

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            continue
        try:
            payload  = json.loads(line)
            X        = parse_payload(payload, features)
            X_scaled = scaler.transform(X)

            pred  = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]
            conf  = max(probs)

            history.append(pred)
            if len(history) >= 3:
                recent = list(history)[-3:]
                if len(set(recent)) == 1:
                    current = recent[0]
                    if current != "unknown" and not detected:
                        elapsed = (datetime.now() - exposure_time).total_seconds()
                        print(f"\n>>> '{current.upper()}' detected in {elapsed:.1f}s from exposure\n")
                        detected = True

            emoji    = EMOJIS.get(current, '❓')
            time_str = datetime.now().strftime('%H:%M:%S')
            prob_str = "  ".join(
                f"{cls}: {p*100:5.1f}%" for cls, p in zip(model.classes_, probs)
            )
            print(f"[{time_str}]  {emoji} {current.upper():<10} ({conf*100:.1f}% conf)  |  {prob_str}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    ser.close()
    print("\nStopped.")


def run_lstm(port, baud, model, scaler, le, features, seq_len, labels):
    import torch

    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected. LSTM needs {seq_len} readings before first prediction.")
    print(f"Press Ctrl+C to stop.\n")

    buffer        = deque(maxlen=seq_len)
    history       = deque(maxlen=5)
    current       = "unknown"
    exposure_time = None
    detected      = False

    input("Place sensor near the smell, then press ENTER to start timing...\n")
    exposure_time = datetime.now()
    print(f"Timer started at {exposure_time.strftime('%H:%M:%S')}")
    print(f"Buffering first {seq_len} readings...\n")

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            continue
        try:
            payload = json.loads(line)
            feat    = parse_payload(payload, features)
            feat_sc = scaler.transform(feat)[0]
            buffer.append(feat_sc)

            time_str = datetime.now().strftime('%H:%M:%S')

            if len(buffer) < seq_len:
                print(f"[{time_str}]  Buffering... ({len(buffer)}/{seq_len})")
                continue

            seq    = torch.FloatTensor(np.array(buffer)).unsqueeze(0)
            with torch.no_grad():
                logits = model(seq)
                probs  = torch.softmax(logits, dim=1).squeeze().numpy()
                pred   = le.inverse_transform([logits.argmax(dim=1).item()])[0]

            conf = max(probs)
            history.append(pred)

            if len(history) >= 3:
                recent = list(history)[-3:]
                if len(set(recent)) == 1:
                    current = recent[0]
                    if current != "unknown" and not detected:
                        elapsed = (datetime.now() - exposure_time).total_seconds()
                        print(f"\n>>> '{current.upper()}' detected in {elapsed:.1f}s from exposure\n")
                        detected = True

            emoji    = EMOJIS.get(current, '❓')
            prob_str = "  ".join(
                f"{cls}: {p*100:5.1f}%" for cls, p in zip(labels, probs)
            )
            print(f"[{time_str}]  {emoji} {current.upper():<10} ({conf*100:.1f}% conf)  |  {prob_str}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    ser.close()
    print("\nStopped.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BME688 eNose Real-time Inference")
    parser.add_argument('--port',  required=True,            help="Serial port e.g. COM8")
    parser.add_argument('--baud',  type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument('--model', default='rf',
                        choices=['rf', 'knn', 'svm', 'mlp', 'ensemble', 'lstm'],
                        help="Model to use (default: rf)")
    args = parser.parse_args()

    print(f"Connecting to {args.port}...")

    if args.model == 'lstm':
        model, scaler, le, features, seq_len, labels = load_lstm_model()
        run_lstm(args.port, args.baud, model, scaler, le, features, seq_len, labels)
    else:
        model, scaler, features = load_sklearn_model(args.model)
        run_sklearn(args.port, args.baud, model, scaler, features)
