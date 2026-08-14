"""
inference.py
------------
Real-time scent classification from BME688 over USB serial.
Arduino can run any MEAS_DUR — the carry-forward buffer decouples the
classification rate from the heater physics (see SensorBuffer below).

Required files in the same folder:
    enose_rf_model.pkl / enose_knn_model.pkl / enose_svm_model.pkl
    enose_mlp_model.pkl / enose_ensemble_model.pkl
    enose_lstm_model.pt
    enose_scaler.pkl
    enose_le.pkl
    feature_columns.pkl

Usage:
    python inference.py --port COM8                          (RF, default stability)
    python inference.py --port COM8 --model lstm
    python inference.py --port COM8 --votes 7 --confidence 0.7   (more stable display)

Stability flags:
    --votes N        consecutive identical predictions needed to change class (default 5)
    --confidence X   minimum confidence 0-1 needed to change class (default 0.6)
    --settle X       safety cap (s) for the adaptive settle-guard (default 6.0) — after a
                     class change, predictions are ignored only until every sensor has
                     actually refreshed since the change, so clean transitions add no delay
    Raise --votes/--confidence if the class flickers (e.g. vinegar briefly showing air).

The output line ends with [N/s] = readings processed per second.
"""

import serial
import json
import numpy as np
import pickle
import joblib
import argparse
from datetime import datetime
from collections import deque

# Optional VR/Unity integration — publishes detected scent to Adafruit IO MQTT.
try:
    from scent_publisher import ScentPublisher
except Exception:
    ScentPublisher = None


# ── Carry-forward sensor buffer ─────────────────────────────────────────────────

class SensorBuffer:
    """
    Maintains the last-known-good reading per sensor.

    In parallel mode each sensor only produces a fresh measurement at the END
    of a heater step. Because the 4 profiles have very different step lengths
    (HP-331 has steps up to ~19s), any poll faster than the slowest step means
    most payloads carry data for only SOME sensors — the rest arrive as null.

    Training data was collected at one full HP-331 cycle per sample (MEAS_DUR
    78400ms), so every training row had ALL 8 sensors populated — a dense
    24-feature vector. To match that distribution at inference time we carry
    forward each sensor's last valid value and overwrite a slot only when a
    fresh reading for that sensor arrives. This reconstructs a dense 8-sensor
    reading on every frame regardless of how fast the Arduino publishes.
    """

    def __init__(self):
        self.gas         = [None] * 8
        self.gas_index   = [None] * 8
        self.temp        = [None] * 8
        self.humidity    = [None] * 8
        self.last_update = [None] * 8  # timestamp each sensor last received a fresh value

    def update(self, payload, now=None):
        """Overwrite a sensor's slot only when the payload carries non-null data.
        `now` (a datetime) is recorded per-sensor so callers can check whether a
        given sensor has refreshed since some reference time (e.g. the last class
        change), instead of guessing a fixed wait duration."""
        if now is None:
            now = datetime.now()
        g  = payload.get('g',  [])
        gi = payload.get('gi', [])
        t  = payload.get('t',  [])
        h  = payload.get('h',  [])
        updated = []
        for i in range(8):
            if i < len(g) and g[i] is not None:
                self.gas[i] = g[i]
                if i < len(gi) and gi[i] is not None:
                    self.gas_index[i] = gi[i]
                if i < len(t) and t[i] is not None:
                    self.temp[i] = t[i]
                if i < len(h) and h[i] is not None:
                    self.humidity[i] = h[i]
                self.last_update[i] = now
                updated.append(i)
        return updated

    def all_refreshed_since(self, ref_time):
        """True once every sensor has received at least one fresh reading at or
        after ref_time. Used to release the settle-guard as soon as the buffer
        is genuinely composed of post-change data, rather than after a fixed
        delay — fast-refreshing sensors release the guard almost immediately,
        while any slower sensor is still respected exactly as long as needed."""
        if ref_time is None:
            return True
        return all(t is not None and t >= ref_time for t in self.last_update)

    def is_warm(self):
        """True once every sensor has reported at least one reading."""
        return all(v is not None for v in self.gas)

    def n_filled(self):
        return sum(1 for v in self.gas if v is not None)

    def build_features(self, features):
        """Build the 24-feature vector from the carried-forward buffer."""
        row = {}
        for i in range(8):
            raw_gas = self.gas[i]       if self.gas[i]       is not None else 0
            gas_idx = self.gas_index[i] if self.gas_index[i] is not None else 0
            row[f'sensor_{i}_gas']       = raw_gas
            row[f'sensor_{i}_gas_index'] = gas_idx
            row[f'sensor_{i}_gas_log']   = np.log1p(raw_gas)
            # Only used if feature_columns includes them (temp/humidity dropped by default)
            row[f'sensor_{i}_temp']      = self.temp[i]     if self.temp[i]     is not None else 0
            row[f'sensor_{i}_humidity']  = self.humidity[i] if self.humidity[i] is not None else 0
        return np.array([[row.get(f, 0) for f in features]])


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


# ── Inference loops ────────────────────────────────────────────────────────────

EMOJIS = {
    'air':     '💨',
    'alcohol': '🥃',
    'coffee':  '☕',
    'vinegar': '🧪',
    'wine':    '🍷'
}


def run_sklearn(port, baud, model, scaler, features, votes, conf_thresh, max_settle_s=6.0, publisher=None):
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected. Running inference — press Ctrl+C to stop.")
    print(f"Stability: {votes} consecutive matching votes + confidence >= {conf_thresh*100:.0f}%\n")

    sensors  = SensorBuffer()
    history  = deque(maxlen=max(votes, 5))
    current  = "unknown"
    detected = False
    last_change_time = None  # timestamp of the most recent class change

    # Readings-per-second counter (rolling 1-second window of arrival times)
    arrivals = deque()

    input("Place sensor near the smell, then press ENTER to start timing...\n")
    exposure_time = datetime.now()
    print(f"Timer started at {exposure_time.strftime('%H:%M:%S')}")
    print("Warming up — waiting for all 8 sensors to report at least once")
    print("(slowest profile HP-331 has ~19s steps, so this can take up to ~20s)\n")

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            continue
        try:
            payload = json.loads(line)
            now = datetime.now()
            sensors.update(payload, now)

            # Update readings/sec — keep only arrivals within the last 1 second
            arrivals.append(now)
            while arrivals and (now - arrivals[0]).total_seconds() > 1.0:
                arrivals.popleft()
            rate = len(arrivals)

            time_str = now.strftime('%H:%M:%S')

            # Wait until every sensor has at least one reading before classifying,
            # so the feature vector is dense and matches the training distribution.
            if not sensors.is_warm():
                print(f"[{time_str}]  Warming up... ({sensors.n_filled()}/8 sensors ready)  [{rate}/s]")
                continue

            X        = sensors.build_features(features)
            X_scaled = scaler.transform(X)

            pred  = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]
            conf  = max(probs)

            # Settle guard: right after a class change, some sensors in the
            # carry-forward buffer may still hold stale values from the
            # PREVIOUS scent while others have already refreshed to the new
            # one. This blended, transitional vector was never seen during
            # training (training rows are always "pure" — collected after a
            # full cycle) and can briefly resemble an unrelated class. Rather
            # than waiting a fixed delay, we check whether EVERY sensor has
            # actually refreshed since the last change — the guard releases
            # the instant the buffer is genuinely post-change data, whether
            # that takes 300ms (fast profile) or several seconds (a slow
            # sensor). `max_settle_s` is only a safety cap in case a sensor
            # stops reporting entirely, so the guard cannot freeze forever.
            settling = (last_change_time is not None and
                        not sensors.all_refreshed_since(last_change_time) and
                        (now - last_change_time).total_seconds() < max_settle_s)

            history.append(pred)
            # Change class only on N consecutive identical predictions AND enough confidence
            if not settling and len(history) >= votes:
                recent = list(history)[-votes:]
                if len(set(recent)) == 1 and conf >= conf_thresh and recent[0] != current:
                    current = recent[0]
                    last_change_time = now
                    history.clear()
                    # Send the confirmed scent to Unity via Adafruit IO MQTT
                    if publisher is not None:
                        publisher.publish(current)
                    if current != "unknown" and not detected:
                        elapsed = (now - exposure_time).total_seconds()
                        print(f"\n>>> '{current.upper()}' detected in {elapsed:.1f}s from exposure\n")
                        detected = True

            status = " (settling)" if settling else ""
            emoji    = EMOJIS.get(current, '❓')
            prob_str = "  ".join(
                f"{cls}: {p*100:5.1f}%" for cls, p in zip(model.classes_, probs)
            )
            print(f"[{time_str}]  {emoji} {current.upper():<10}{status} ({conf*100:.1f}% conf)  |  {prob_str}  [{rate}/s]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    ser.close()
    print("\nStopped.")


def run_lstm(port, baud, model, scaler, le, features, seq_len, labels, votes, conf_thresh, max_settle_s=6.0, publisher=None):
    import torch

    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected. LSTM needs {seq_len} readings after warm-up.")
    print(f"Stability: {votes} consecutive matching votes + confidence >= {conf_thresh*100:.0f}%")
    print(f"Press Ctrl+C to stop.\n")

    sensors    = SensorBuffer()
    seq_buffer = deque(maxlen=seq_len)
    history    = deque(maxlen=max(votes, 5))
    current    = "unknown"
    detected   = False
    last_change_time = None

    arrivals = deque()

    input("Place sensor near the smell, then press ENTER to start timing...\n")
    exposure_time = datetime.now()
    print(f"Timer started at {exposure_time.strftime('%H:%M:%S')}")
    print("Warming up all 8 sensors, then buffering the sequence\n")

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            continue
        try:
            payload = json.loads(line)
            now = datetime.now()
            sensors.update(payload, now)

            arrivals.append(now)
            while arrivals and (now - arrivals[0]).total_seconds() > 1.0:
                arrivals.popleft()
            rate = len(arrivals)

            time_str = now.strftime('%H:%M:%S')

            if not sensors.is_warm():
                print(f"[{time_str}]  Warming up... ({sensors.n_filled()}/8 sensors ready)  [{rate}/s]")
                continue

            feat    = sensors.build_features(features)
            feat_sc = scaler.transform(feat)[0]
            seq_buffer.append(feat_sc)

            if len(seq_buffer) < seq_len:
                print(f"[{time_str}]  Buffering sequence... ({len(seq_buffer)}/{seq_len})  [{rate}/s]")
                continue

            seq = torch.FloatTensor(np.array(seq_buffer)).unsqueeze(0)
            with torch.no_grad():
                logits = model(seq)
                probs  = torch.softmax(logits, dim=1).squeeze().numpy()
                pred   = le.inverse_transform([logits.argmax(dim=1).item()])[0]

            conf = max(probs)

            # Settle guard — see explanation in run_sklearn. Releases as soon as
            # every sensor has genuinely refreshed since the last class change,
            # rather than after a fixed delay.
            settling = (last_change_time is not None and
                        not sensors.all_refreshed_since(last_change_time) and
                        (now - last_change_time).total_seconds() < max_settle_s)

            history.append(pred)

            if not settling and len(history) >= votes:
                recent = list(history)[-votes:]
                if len(set(recent)) == 1 and conf >= conf_thresh and recent[0] != current:
                    current = recent[0]
                    last_change_time = now
                    history.clear()
                    # Send the confirmed scent to Unity via Adafruit IO MQTT
                    if publisher is not None:
                        publisher.publish(current)
                    if current != "unknown" and not detected:
                        elapsed = (now - exposure_time).total_seconds()
                        print(f"\n>>> '{current.upper()}' detected in {elapsed:.1f}s from exposure\n")
                        detected = True

            status = " (settling)" if settling else ""
            emoji    = EMOJIS.get(current, '❓')
            prob_str = "  ".join(
                f"{cls}: {p*100:5.1f}%" for cls, p in zip(labels, probs)
            )
            print(f"[{time_str}]  {emoji} {current.upper():<10}{status} ({conf*100:.1f}% conf)  |  {prob_str}  [{rate}/s]")

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
    parser.add_argument('--votes', type=int, default=5,
                        help="Consecutive matching predictions required to change class (default 5). "
                             "Higher = more stable, slower to switch. At 200ms, 5 votes = 1s of agreement.")
    parser.add_argument('--confidence', type=float, default=0.7,
                        help="Minimum confidence (0-1) required to change class (default 0.6). "
                             "Higher = ignores borderline predictions.")
    parser.add_argument('--settle', type=float, default=6.0,
                        help="Safety cap in seconds for the settle-guard (default 6.0). "
                             "After any class change, predictions are ignored until EVERY sensor "
                             "has refreshed at least once since the change — this releases almost "
                             "immediately for fast-refreshing sensors and adds no delay to clean "
                             "transitions. This value only caps how long the guard can hold if a "
                             "sensor stops refreshing, so the display can never freeze indefinitely.")
    parser.add_argument('--vr', action='store_true',
                        help="Enable VR/Unity integration: publish detected scents to Adafruit IO MQTT.")
    parser.add_argument('--aio-user', default='alen27', help="Adafruit IO username (with --vr)")
    parser.add_argument('--aio-key',  default='aio_ecDT20RsfPc49R4fYhsdsXHNlGwg', help="Adafruit IO key (with --vr)")
    parser.add_argument('--aio-feed', default='enose', help="Adafruit IO feed name (with --vr)")
    args = parser.parse_args()

    publisher = None
    if args.vr:
        if ScentPublisher is None:
            print("ERROR: --vr requires paho-mqtt and scent_publisher.py in the same folder.")
            print("Install with: python -m pip install paho-mqtt")
        else:
            publisher = ScentPublisher(args.aio_user, args.aio_key, args.aio_feed)

    print(f"Connecting to {args.port}...")

    if args.model == 'lstm':
        model, scaler, le, features, seq_len, labels = load_lstm_model()
        run_lstm(args.port, args.baud, model, scaler, le, features, seq_len, labels,
                 args.votes, args.confidence, args.settle, publisher)
    else:
        model, scaler, features = load_sklearn_model(args.model)
        run_sklearn(args.port, args.baud, model, scaler, features,
                    args.votes, args.confidence, args.settle, publisher)