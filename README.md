# AI-Driven Olfaction for Real-Time Scent Detection in Extended Reality Systems

**Project repository — EEN1095 Final Portfolio**

| **Author** | Alen Skaria Sabu (21106487) |
| **School** | School of Electronic Engineering, Dublin City University |
| **Supervisor** | Prof. Gabriel-Miro Muntean |
| **Advisor** | Dr. Anderson Augusto Simiscuka |
| **Date** | August 2026 |

---

## 1. What this repository contains

This repository holds the firmware, software, datasets, trained models and XR
integration for an electronic nose (eNose) that classifies real-world scents in
real time and delivers the result to a Unity application running on a Meta
Quest 2 headset.

The system detects ten scent classes — ambient air, four liquids and five ground
spices — using a Bosch BME688 eight-sensor array driven by an ESP32
microcontroller. A soft-voting ensemble attains 96.4% accuracy under blocked
temporal partitioning, with 13.8 ms inference latency.

Everything required to reproduce the results reported in the portfolio and the
conference paper is contained here. No external cloud links are required.

---

## 2. Core items for detailed review

The ten items below are proposed for detailed assessment. Section 4 explains the
full directory structure, and Section 6 gives instructions for running each stage.

| # | Item | Description |
|---|------|-------------|
| 1 | `README.md` | This file — repository structure, core items and operating instructions. |
| 2 | `ARDUINO CODE/bme688_dev_kit` | ESP32 firmware. Configures all eight sensors with heater profile HP-354, reads them in parallel mode, and emits JSON over serial. Contains the `TRAINING_MODE` flag selecting the acquisition and inference regimes. |
| 3 | `collect_data.py` | Labelled data acquisition over serial. Parses each JSON payload, expands it to one row per reading with all eight sensors as separate columns, and writes a per-class CSV. |
| 4 | `Training_Colab_any_class.ipynb` | Training notebook. Feature engineering, blocked temporal partitioning, training and evaluation of all six classifiers, and export of the model artefacts. Contains the stored outputs from which every figure in the portfolio is derived. |
| 5 | `models/inference.py` | Real-time classification. Implements the carry-forward buffer (Algorithm 1) and the adaptive settle guard (Algorithm 2), model selection across all six classifiers, and the optional MQTT publisher for XR delivery. |
| 6 | `models/scent_publisher.py` | Adafruit IO MQTT publisher used by `inference.py` to deliver confirmed detections to the Unity client. |
| 7 | `*_data.csv` (10 files) | The ten labelled datasets, 8572 records in total, approximately 858 per class. |
| 8 | `models/` (9 artefacts) | Trained models, scaler, label encoder and feature-column definition required by `inference.py`. |
| 9 | `Unity Application` | Unity MQTT subscriber. Extends the supplied `M2MqttUnityClient` base class, performs thread-safe handoff from the network thread to the rendering thread, and maps class names to scene objects. |
| 10 | `demo/demo_video.mp4` | Demonstration of the working system: a scent presented to the array, and the corresponding object revealed in the Meta Quest 2 headset. |

---

## 3. Requirements

**Hardware**

- Bosch BME688 development kit, populated with eight BME688 sensors
- Adafruit HUZZAH32 (ESP32 Feather) board
- USB cable to host computer
- Meta Quest 2 headset with Quest Link cable (XR stage only)

**Software**

- Arduino IDE with ESP32 board support and the `bme68xLibrary`
- Python 3.10 or later
- Unity 2021.3 LTS or later, with the Oculus XR plugin
- A Google account for Colab, if the training notebook is to be re-run

**Python packages**

```
pip install pyserial pandas numpy scikit-learn==1.6.1 torch joblib paho-mqtt
```

The scikit-learn version matters. The models were trained under 1.6.1; loading
them under a later version raises an `InconsistentVersionWarning`. The warning is
generally harmless but pinning the version removes it and guarantees identical
behaviour.

---

## 4. Directory structure

```
21106487_project_repository/
│
├── README.md                          ← this file
│
├── ARDUINO CODE/
│   └── bme688_dev_kit/
│       ├── bme688_dev_kit.ino         ← earlier variant with MQTT (superseded)
│       ├── commMux.cpp / commMux.h    ← multiplexer driver (vendor-supplied)
│ 
│
├── collect_data.py                    ← CORE ITEM 3: acquisition utility
├── Training_Colab_any_class.ipynb     ← CORE ITEM 4: training and evaluation
│
├── air_data.csv                       ← CORE ITEM 7: the ten datasets
├── chillipowder_data.csv
├── cinnamon_data.csv
├── coffee_data.csv
├── corrianderpowder_data.csv
├── masalapowder_data.csv
├── pepperpowder_data.csv
├── vinegar_data.csv
├── whiskey_data.csv
├── wine_data.csv
│
├── models/                            ← CORE ITEM 8: trained artefacts
│   ├── inference.py                   ← CORE ITEM 5: real-time classification
│   ├── scent_publisher.py             ← CORE ITEM 6: MQTT publisher
│   ├── enose_knn_model.pkl
│   ├── enose_rf_model.pkl
│   ├── enose_svm_model.pkl
│   ├── enose_mlp_model.pkl
│   ├── enose_ensemble_model.pkl
│   ├── enose_lstm_model.pt
│   ├── enose_scaler.pkl               ← MinMaxScaler fitted on training data only
│   ├── enose_le.pkl                   ← LabelEncoder for the neural models
│   └── feature_columns.pkl            ← the 24 feature names, in order
│
├── unity/
│   └── ScentReceiver.cs               ← CORE ITEM 9: Unity MQTT subscriber
│
└── demo/
    └── demo_video.mp4                 ← CORE ITEM 10: system demonstration
```

**Supplementary material.** `ARDUINO CODE/bme688_dev_kit/bme688_dev_kit.ino` is an
earlier firmware variant that published directly to MQTT from the microcontroller.
It was superseded by the serial-only version once acquisition was moved off the
network, and is retained for completeness rather than proposed for review. Any
`__pycache__` directories are compilation artefacts and may be ignored.

---

## 5. Dataset description

Ten classes, balanced at approximately 858 records each, 8572 records in total.

| Class | Category | Records |
|-------|----------|---------|
| air | baseline | 858 |
| vinegar | liquid | 858 |
| whiskey | liquid | 858 |
| wine | liquid | 858 |
| coffee | liquid | 854 |
| chillipowder | spice | 854 |
| cinnamon | spice | 858 |
| corrianderpowder | spice | 858 |
| masalapowder | spice | 858 |
| pepperpowder | spice | 858 |

**CSV columns.** Each file contains `timestamp`, `received_time`, `label`, and
then four columns per sensor for each of the eight sensors:

- `sensor_N_gas` — gas resistance in ohms
- `sensor_N_gas_index` — heater step (0–9) at which the reading was taken
- `sensor_N_temp` — package temperature (recorded but not used as a feature)
- `sensor_N_humidity` — relative humidity (recorded but not used as a feature)

Temperature and humidity are excluded from the feature set because both are
measured within the sensor package and distorted by self-heating, so they partly
encode elapsed operating time rather than any property of the scent. The
reasoning and the empirical check are given in Appendix D.5 of the portfolio.

**Acquisition conditions.** Each class was recorded in a single continuous
session of approximately 300 s at a 350 ms transmission interval, following a
warm-up interval allowing the sensing layer to stabilise, with a neutralisation
interval in fresh air between classes.

---

## 6. Running the pipeline

The three stages are independent. Stage B may be run on the supplied datasets
without any hardware.

### Stage A — Data acquisition (requires hardware)

1. Open `bme688_dev_kit.ino` in the Arduino IDE.
2. Confirm the acquisition regime is selected:
   ```cpp
   #define TRAINING_MODE true    // 350 ms transmission interval
   ```
3. Flash to the HUZZAH32. Close the Serial Monitor afterwards — it holds the
   port open and `collect_data.py` will fail with a permission error if it is
   left running.
4. Collect one class at a time:
   ```
   python collect_data.py --port COM3 --label coffee --duration 300
   ```
   The script prompts for a save location on completion. Pressing Enter writes
   `coffee_data.csv` to the current folder.

Allow the array to warm up before starting, and return it to open air between
classes.

### Stage B — Training and evaluation (no hardware required)

1. Upload `Training_Colab_any_class.ipynb` and the ten CSV files to Colab.
2. Run all cells in order.

The notebook loads whatever `*_data.csv` files are present, so the class count is
not fixed in code. It performs feature engineering into the 24-dimensional
representation, applies blocked temporal partitioning, trains all six
classifiers, and writes the nine model artefacts.

**On the evaluation protocol.** Partitioning is by contiguous temporal blocks
rather than at random. Records are captured at 350 ms while heater steps last
between 280 ms and 4200 ms, so a record and its immediate neighbours frequently
share the same underlying measurement on most sensors; a random split would
distribute such neighbours across both partitions and measure memorisation
rather than generalisation. The first 80% of each class recording trains and the
final 20% tests. For the recurrent model, sliding windows are constructed
separately within each partition so that no sequence straddles a boundary, and a
further 20% of the training sequences is withheld for epoch selection.

3. Download the nine artefacts from `models/` in the Colab file browser into the
   local `models/` folder.

### Stage C — Real-time inference

1. Reflash the firmware with the inference regime selected:
   ```cpp
   #define TRAINING_MODE false   // 50 ms transmission interval
   ```
2. Run from within the `models/` folder:
   ```
   python inference.py --port COM3 --model ensemble
   ```

**Command-line options**

| Flag | Default | Meaning |
|------|---------|---------|
| `--port` | required | Serial port, e.g. `COM3` |
| `--baud` | 115200 | Baud rate |
| `--model` | `rf` | One of `knn`, `rf`, `svm`, `mlp`, `ensemble`, `lstm` |
| `--votes` | 5 | Consecutive agreeing predictions required to change class |
| `--confidence` | 0.7 | Minimum confidence required to change class |
| `--settle` | 6.0 | Safety ceiling in seconds for the adaptive settle guard |
| `--vr` | off | Publish confirmed detections to Adafruit IO for the Unity client |
| `--aio-user`, `--aio-key`, `--aio-feed` | — | Adafruit IO credentials, used with `--vr` |

Raise `--votes` or `--confidence` if the reported class flickers. The `--settle`
value is only a ceiling: the guard normally releases as soon as every sensor has
refreshed since the last class change, which under the uniform HP-354 profile
adds no measurable delay.

### Stage D — XR integration

1. Copy `unity/ScentReceiver.cs` into `Assets/Olfaction Scripts/` in the Unity
   project.
2. Create an empty GameObject named `ScentReceiver` and attach the script.
3. In the Inspector, set the Adafruit IO username, key and feed name. These must
   match the values passed to `inference.py`.
4. Populate the **Scent Objects** list, mapping each class name to a scene
   object. Class names must match the training labels exactly, in lower case.
5. Connect the Meta Quest 2 by Quest Link, press Play in Unity, and run
   `inference.py` with `--vr`.

All mapped objects are deactivated on start, so the scene begins empty. Receipt
of the `air` class clears the scene rather than revealing an object.

Verify operation through the Unity Console: a line reading
`[ScentReceiver] Received: coffee` confirms the MQTT round trip. If that line
appears but no object is revealed, the cause is object positioning or a mismatch
between the class name and the Inspector entry, not the network path.

---

## 7. Reproducing the reported results

The figures reported in the portfolio and paper are contained in the stored
outputs of `Training_Colab_any_class.ipynb` and may be read without re-running it.

| Model | Accuracy | Macro F1 | Blocked CV | Latency |
|-------|----------|----------|------------|---------|
| k-Nearest Neighbours | 95.1% | 95.0% | 95.5% ± 3.5 | 1.52 ms |
| Random Forest | 91.0% | 90.8% | 93.8% ± 6.0 | 30.52 ms |
| Support Vector Machine | 93.6% | 93.4% | 93.3% ± 3.3 | 0.38 ms |
| Multilayer Perceptron | 91.4% | 91.5% | 92.7% ± 3.5 | 0.39 ms |
| Long Short-Term Memory | 90.7% | 90.5% | — | 0.28 ms |
| **Soft-Voting Ensemble** | **96.4%** | **96.3%** | **96.3% ± 3.3** | 13.84 ms |

The recurrent model has no cross-validation figure because it uses a held-out
validation split for epoch selection rather than k-fold resampling; reporting its
validation accuracy in that column would not be comparable with the other five.

**Expected variation on reproduction.** Two sources should be anticipated. The
absolute resistance of a metal-oxide layer depends on the individual device, its
accumulated operating history and ambient conditions, so absolute values will
differ between devices; the relative separability of the classes, on which
classification depends, is the quantity that should transfer. Second, results
depend on the partitioning strategy, so a reproduction adopting a random split
should be expected to report a higher figure without any underlying difference in
capability.

Latency figures were measured on the host computer, not an embedded target.
Absolute values would be substantially larger on a microcontroller, though the
ordering across model families would be preserved.

---

## 8. Known issues and notes

**Credentials are hardcoded.** `inference.py` carries an Adafruit IO username and
key as argument defaults. These are present so the demonstration runs without
additional configuration, and the feed carries only scent class names. They
should be replaced before any wider use.

**Serial port contention.** Only one program may hold the port. Close the Arduino
Serial Monitor before running `collect_data.py` or `inference.py`, or the script
will fail with a permission error.

**Ambient sensitivity.** Classification accuracy degrades when ambient conditions
differ appreciably from those at acquisition. Two failure modes were observed:
ambient air intermittently reported as vinegar, and presented scents reported as
air. Both are consistent with a shift in the absolute resistance scale relative
to training. This is documented in Appendix E.10 of the portfolio, together with
the two candidate mitigations.

---

## 9. Where to find further detail

| Topic | Location |
|-------|----------|
| Design decisions and rejected alternatives | Portfolio, Appendix D |
| Heater profile derivation and step parameters | Portfolio, Appendix D.2 |
| Evaluation protocol in full | Portfolio, Appendix E.2 |
| Per-class results across all six models | Portfolio, Appendix E.4 |
| Outstanding testing and limitations | Portfolio, Appendix E.10 |
| Use of generative AI | Portfolio, Appendix A |
