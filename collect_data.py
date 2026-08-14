"""
collect_data.py
---------------
Run this in VS Code to collect labelled training data from the BME688 over USB.
Make sure the Arduino is flashed with TRAINING_MODE = true before running.

Cycle-matched collection: with the single HP-354 profile (10.78s per full cycle),
MEAS_DUR on the Arduino should be 10780ms so every saved row has all 8 sensors
genuinely fresh. At that rate ~835 samples takes about 2.5 hours (--duration 9500).

Usage:
    python collect_data.py --port COM8 --label air      --duration 9500
    python collect_data.py --port COM8 --label vinegar  --duration 9500
    python collect_data.py --port COM8 --label whiskey  --duration 9500
    python collect_data.py --port COM8 --label chillipowder --duration 9500

After collection finishes you will be prompted for where to save the CSV.
Press ENTER at that prompt to save in the current folder, or type a path
(folder or full file path) to save elsewhere.
"""

import serial
import json
import pandas as pd
from datetime import datetime
import argparse
import os


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
    # ~10.78s per genuinely fresh sample at the HP-354 cycle rate
    est_samples = int(duration_seconds // 10.78)
    print(f"Collecting: '{label}' for {duration_seconds}s (~{est_samples} samples at 10.78s/cycle)")
    print(f"{'='*50}\n")

    buffer = []
    start  = datetime.now()

    # total_seconds() rather than .seconds — .seconds alone drops the 'days'
    # component and would be wrong for any collection run over 24 hours.
    while (datetime.now() - start).total_seconds() < duration_seconds:
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


def prompt_save_path(default_filename):
    """
    Ask the user where to save the CSV. Accepts:
      - blank (ENTER)      -> save default_filename in the current folder
      - a folder path      -> save default_filename inside that folder
      - a full file path   -> save exactly there (creates parent folders if needed)
    """
    print(f"\nWhere would you like to save the file?")
    print(f"  Press ENTER to save as '{default_filename}' in the current folder,")
    print(f"  or type a folder path, or a full file path ending in .csv")
    choice = input("Save location: ").strip().strip('"')

    if not choice:
        return default_filename

    # If it looks like an existing folder (or has no .csv extension), treat as a folder
    if os.path.isdir(choice) or not choice.lower().endswith('.csv'):
        os.makedirs(choice, exist_ok=True)
        return os.path.join(choice, default_filename)

    # Otherwise treat as a full file path — make sure the parent folder exists
    parent = os.path.dirname(choice)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return choice


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

    df = pd.DataFrame(records)

    default_filename = f"{label}_data.csv"
    save_path = prompt_save_path(default_filename)

    df.to_csv(save_path, index=False)
    print(f"\nSaved to {os.path.abspath(save_path)}  (shape: {df.shape})")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BME688 eNose Data Collector")
    parser.add_argument('--port',     required=True,           help="Serial port e.g. COM8")
    parser.add_argument('--label',    required=True,           help="Scent label e.g. air, chillipowder")
    parser.add_argument('--duration', type=int, default=9500,  help="Collection duration in seconds (default 9500, ~2.5h at 10.78s cycle)")
    parser.add_argument('--baud',     type=int, default=115200, help="Baud rate (default 115200)")
    args = parser.parse_args()

    buffer = collect(args.port, args.baud, args.label, args.duration)
    save(buffer, args.label)