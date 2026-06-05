"""
collect_data.py
---------------
Run this in VS Code to collect labelled training data from the BME688 over USB.
Make sure the Arduino is flashed with TRAINING_MODE = true before running.

Usage:
    python collect_data.py --port COM8 --label air     --duration 63200
    python collect_data.py --port COM8 --label coffee  --duration 63200
    python collect_data.py --port COM8 --label alcohol --duration 63200
    python collect_data.py --port COM8 --label vinegar --duration 63200
    python collect_data.py --port COM8 --label wine    --duration 63200
"""

import serial
import json
import pandas as pd
from datetime import datetime
import argparse


def parse_payload(payload):
    gas   = payload.get('g',  [])
    gi    = payload.get('gi', [])
    temp  = payload.get('t',  [])
    hum   = payload.get('h',  [])
    sensors = []
    for i in range(8):
        sensors.append({
            'id':           i,
            'gas_resistance': gas[i]  if i < len(gas)  and gas[i]  is not None else None,
            'gas_index':      gi[i]   if i < len(gi)   and gi[i]   is not None else None,
            'temperature':    temp[i] if i < len(temp) and temp[i] is not None else None,
            'humidity':       hum[i]  if i < len(hum)  and hum[i]  is not None else None,
        })
    return {'timestamp': payload.get('ts'), 'sensors': sensors}


def collect(port, baud, label, duration_seconds):
    print(f"\nConnecting to {port}...")
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected.")
    print(f"\n{'='*50}")
    print(f"Collecting: '{label}' for {duration_seconds}s (~{duration_seconds//79} samples)")
    print(f"{'='*50}\n")

    buffer = []
    start  = datetime.now()

    while (datetime.now() - start).seconds < duration_seconds:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            continue
        try:
            payload  = json.loads(line)
            expanded = parse_payload(payload)
            expanded['received_time'] = datetime.now().isoformat()
            expanded['label']         = label
            buffer.append(expanded)
            print(f"Samples: {len(buffer)}", end='\r')
        except Exception as e:
            print(f"\nParse error: {e} | Line: {line[:80]}")

    ser.close()
    print(f"\nCollection complete. Samples: {len(buffer)}")
    return buffer


def save(buffer, label):
    if not buffer:
        print("No data to save!")
        return

    records = []
    for entry in buffer:
        record = {
            'timestamp':     entry['timestamp'],
            'received_time': entry['received_time'],
            'label':         entry['label']
        }
        for sensor in entry['sensors']:
            sid = sensor['id']
            record[f'sensor_{sid}_gas']       = sensor['gas_resistance']
            record[f'sensor_{sid}_gas_index'] = sensor['gas_index']
            record[f'sensor_{sid}_temp']      = sensor['temperature']
            record[f'sensor_{sid}_humidity']  = sensor['humidity']
        records.append(record)

    df       = pd.DataFrame(records)
    filename = f"{label}_data.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}  (shape: {df.shape})")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BME688 eNose Data Collector")
    parser.add_argument('--port',     required=True,           help="Serial port e.g. COM8")
    parser.add_argument('--label',    required=True,           help="Scent label e.g. air, coffee")
    parser.add_argument('--duration', type=int, default=63200, help="Collection duration in seconds (default 63200)")
    parser.add_argument('--baud',     type=int, default=115200,help="Baud rate (default 115200)")
    args = parser.parse_args()

    buffer = collect(args.port, args.baud, args.label, args.duration)
    save(buffer, args.label)
