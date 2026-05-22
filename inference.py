"""
inference.py
------------
Real-time scent classification from BME688 over USB serial.
Arduino must be flashed with TRAINING_MODE = true.

Make sure these files are in the same folder:
    enose_rf_model.pkl
    enose_knn_model.pkl
    enose_svm_model.pkl
    enose_ensemble_model.pkl
    enose_scaler.pkl
    feature_columns.pkl

Usage:
    python inference.py --port COM8              (RF by default)
    python inference.py --port COM8 --model knn
    python inference.py --port COM8 --model svm
    python inference.py --port COM8 --model ensemble
"""

import serial
import json
import numpy as np
import pickle
import joblib
import argparse
from datetime import datetime
from collections import deque


def load_model(model_type='rf'):
    try:
        model_file = f'enose_{model_type}_model.pkl'
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open('enose_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        features = joblib.load('feature_columns.pkl')

        print(f"Model loaded: {model_file}")
        print(f"Classes:  {list(model.classes_)}")
        print(f"Features: {len(features)} features\n")
        return model, scaler, features
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the .pkl files are in the same folder as this script.")
        raise


def parse_payload(payload, features):
    gas  = payload.get('g', [])
    temp = payload.get('t', [])
    hum  = payload.get('h', [])

    row = {}
    for i in range(8):
        raw_gas = gas[i] if i < len(gas) and gas[i] is not None else 0
        row[f'sensor_{i}_gas']      = raw_gas
        row[f'sensor_{i}_gas_log']  = np.log1p(raw_gas)
        row[f'sensor_{i}_temp']     = temp[i] if i < len(temp) and temp[i] is not None else 0
        row[f'sensor_{i}_humidity'] = hum[i]  if i < len(hum)  and hum[i]  is not None else 0

    return np.array([[row.get(f, 0) for f in features]])


EMOJIS = {
    'air':     '💨',
    'alcohol': '🥃',
    'coffee':  '☕',
    'vinegar': '🧪',
    'wine':    '🍷'
}


def run(port, baud, model, scaler, features):
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

            # Majority vote over last 3 predictions
            if len(history) >= 3:
                recent = list(history)[-3:]
                if len(set(recent)) == 1:
                    current = recent[0]

                    # Record end-to-end latency on first confirmed detection
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BME688 eNose Real-time Inference")
    parser.add_argument('--port',  required=True,            help="Serial port e.g. COM8 or /dev/ttyUSB0")
    parser.add_argument('--baud',  type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument('--model', default='rf',
                        choices=['rf', 'knn', 'svm', 'mlp', 'ensemble'],
                        help="Model to use (default: rf)")
    args = parser.parse_args()

    model, scaler, features = load_model(args.model)
    run(args.port, args.baud, model, scaler, features)
