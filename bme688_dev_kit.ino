/*
 * eNose - Custom Implementation (Serial-only, no WiFi/MQTT)
 * Hardware: BME688 Evaluation Kit + ESP32 Feather
 *
 * SINGLE heater profile: HP-354 (10.78s) on ALL 8 sensors.
 *   All sensors: HP-354 — Air, Coffee, Vinegar, Alcohol, Wine, Spices
 *
 * Outputs one JSON line per reading over USB serial. No cloud, no network.
 * Both collect_data.py and inference.py read this serial stream directly.
 *
 * TRAINING_MODE true  -> MEAS_DUR 10780ms (one full HP-354 cycle per sample,
 *                        dense rows for data collection with collect_data.py)
 * TRAINING_MODE false -> MEAS_DUR 50ms (fast updates for real-time inference;
 *                        inference.py's carry-forward buffer fills the gaps)
 */

#include <ArduinoJson.h>
#include "bme68xLibrary.h"
#include "commMux.h"

// ==================== CONFIGURATION ====================

#define TRAINING_MODE true   // true = collect data (10780ms) | false = inference (50ms)

#if TRAINING_MODE
  #define MEAS_DUR 350      // one full HP-354 cycle (10.78s) -> dense rows
#else
  #define MEAS_DUR 50        // fast inference; carry-forward buffer handles gaps
#endif

#define N_KIT_SENS 8

// ==================== HEATER PROFILE ====================

// HP-354 (10.78s) — used on ALL 8 sensors
uint16_t tempHP354[10] = {320, 100, 100, 100, 200, 200, 200, 320, 320, 320};
uint16_t mulHP354[10]  = {5,   2,   10,  30,  5,   5,   5,   5,   5,   5};

// ==================== GLOBAL OBJECTS ====================

unsigned long lastSend = 0;

Bme68x bme[N_KIT_SENS];
commMux commSetup[N_KIT_SENS];

// ==================== SETUP ====================

void setup() {
  Serial.begin(115200);
  delay(1000);

  #if TRAINING_MODE
    Serial.println("\n=== eNose Starting (TRAINING MODE - serial) ===");
  #else
    Serial.println("\n=== eNose Starting (INFERENCE MODE - serial) ===");
  #endif

  Serial.println("Profile: HP-354 on all 8 sensors");

  commMuxBegin(Wire, SPI);
  delay(100);

  initSensors();
  Serial.println("Ready!");
}

// ==================== LOOP ====================

void loop() {
  if (millis() - lastSend >= MEAS_DUR) {
    lastSend = millis();

    uint32_t gas[8]   = {0};
    uint8_t  gidx[8]  = {0};
    float    temp[8]  = {0};
    float    hum[8]   = {0};
    bool     valid[8] = {false};

    for (uint8_t i = 0; i < N_KIT_SENS; i++) {
      if (bme[i].fetchData()) {
        bme68xData data;
        bme[i].getData(data);
        if (data.status & BME68X_NEW_DATA_MSK) {
          gas[i]   = data.gas_resistance;
          gidx[i]  = data.gas_index;
          temp[i]  = data.temperature;
          hum[i]   = data.humidity;
          valid[i] = true;
        }
      }
    }

    StaticJsonDocument<512> doc;
    doc["ts"]  = millis();
    doc["lbl"] = "unknown";

    JsonArray g  = doc.createNestedArray("g");
    JsonArray gi = doc.createNestedArray("gi");
    JsonArray t  = doc.createNestedArray("t");
    JsonArray h  = doc.createNestedArray("h");

    bool anyValid = false;
    for (uint8_t i = 0; i < N_KIT_SENS; i++) {
      if (valid[i]) {
        g.add(gas[i]);
        gi.add(gidx[i]);
        t.add(round(temp[i] * 100) / 100.0);
        h.add(round(hum[i] * 100) / 100.0);
        anyValid = true;
      } else {
        g.add(nullptr);
        gi.add(nullptr);
        t.add(nullptr);
        h.add(nullptr);
      }
    }

    if (!anyValid) {
      // Nothing fresh this cycle — stay silent so the JSON stream stays clean
      return;
    }

    char payload[512];
    serializeJson(doc, payload);
    Serial.println(payload);   // one JSON line per reading
  }
}

// ==================== HELPERS ====================

void initSensors() {
  Serial.print("Sensors: ");
  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    commSetup[i] = commMuxSetConfig(Wire, SPI, i, commSetup[i]);
    bme[i].begin(BME68X_SPI_INTF, commMuxRead, commMuxWrite, commMuxDelay, &commSetup[i]);

    if (!bme[i].checkStatus()) {
      bme[i].setTPH();
      uint16_t sharedHeatrDur = MEAS_DUR - (bme[i].getMeasDur(BME68X_PARALLEL_MODE) / INT64_C(1000));
      bme[i].setHeaterProf(tempHP354, mulHP354, sharedHeatrDur, 10);   // HP-354 on every sensor
      bme[i].setOpMode(BME68X_PARALLEL_MODE);
      Serial.print(i);
      Serial.print(" ");
    } else {
      Serial.print("X ");
    }
  }
  Serial.println();
}
