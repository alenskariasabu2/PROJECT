/*
 * eNose - Custom Implementation
 * Hardware: BME688 Evaluation Kit + ESP32 Feather
 *
 * TRAINING_MODE true  -> outputs JSON over Serial at 500ms, no WiFi/MQTT needed
 * TRAINING_MODE false -> sends JSON to Adafruit IO via MQTT every 2 seconds
 */

#include <ArduinoJson.h>
#include "bme68xLibrary.h"
#include "commMux.h"

// ==================== CONFIGURATION ====================

#define TRAINING_MODE true   // <-- set to false before demo

#if TRAINING_MODE
  #define MEAS_DUR 500       // 0.5 seconds during training
#else
  #define MEAS_DUR 2000      // 2 seconds during demo
  #include <WiFi.h>
  #include <Adafruit_MQTT.h>
  #include <Adafruit_MQTT_Client.h>

  // WiFi Credentials
  #define WIFI_SSID "VODAFONE-EBC8"
  #define WIFI_PASS "HfG47PKMNa4cAR9n"

  // Adafruit IO Credentials
  #define AIO_SERVER      "io.adafruit.com"
  #define AIO_SERVERPORT  1883
  #define AIO_USERNAME    "alen27"
  #define AIO_KEY         "aio_YKru23fPsPZRaPCnSLHCe5QRVU2L"

  #define MQTT_TOPIC AIO_USERNAME "/feeds/enose"
#endif

// Hardware Configuration
#define N_KIT_SENS 8

// ==================== GLOBAL OBJECTS ====================

unsigned long lastSend = 0;

#if !TRAINING_MODE
  WiFiClient client;
  Adafruit_MQTT_Client mqtt(&client, AIO_SERVER, AIO_SERVERPORT, AIO_USERNAME, AIO_KEY);
  Adafruit_MQTT_Publish sensorFeed = Adafruit_MQTT_Publish(&mqtt, MQTT_TOPIC);
#endif

Bme68x bme[N_KIT_SENS];
commMux commSetup[N_KIT_SENS];

uint16_t tempProf[10] = {320, 100, 100, 100, 200, 200, 200, 320, 320, 320};
uint16_t mulProf[10]  = {5, 2, 10, 30, 5, 5, 5, 5, 5, 5};

// ==================== SETUP ====================

void setup() {
  Serial.begin(115200);
  delay(1000);

  #if TRAINING_MODE
    Serial.println("\n=== eNose Starting (TRAINING MODE) ===");
  #else
    Serial.println("\n=== eNose Starting (DEMO MODE) ===");
  #endif

  commMuxBegin(Wire, SPI);
  delay(100);

  #if !TRAINING_MODE
    connectWiFi();
  #endif

  initSensors();
  Serial.println("Ready!");
}

// ==================== LOOP ====================

void loop() {
  #if !TRAINING_MODE
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi lost, reconnecting...");
      connectWiFi();
    }
    MQTT_connect();
  #endif

  if (millis() - lastSend >= MEAS_DUR) {
    lastSend = millis();

    uint32_t gas[8]  = {0};
    uint8_t  gidx[8] = {0};
    float    temp[8] = {0};
    float    hum[8]  = {0};
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
      Serial.println("No valid sensor data, skipping");
      return;
    }

    char payload[512];
    serializeJson(doc, payload);

    #if TRAINING_MODE
      // In training mode: print JSON on its own line for Python to parse
      Serial.println(payload);
    #else
      // In demo mode: publish to Adafruit IO
      Serial.print("Sending: ");
      Serial.println(payload);
      if (!sensorFeed.publish(payload)) {
        Serial.println("MQTT FAIL");
      } else {
        Serial.println("MQTT OK");
      }
    #endif
  }
}

// ==================== HELPERS ====================

#if !TRAINING_MODE
void connectWiFi() {
  Serial.print("WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("OK");
  Serial.println(WiFi.localIP());
}

void MQTT_connect() {
  if (mqtt.connected()) return;
  Serial.print("MQTT...");
  while (mqtt.connect() != 0) {
    Serial.print(".");
    mqtt.disconnect();
    delay(5000);
  }
  Serial.println("OK");
}
#endif

void initSensors() {
  Serial.print("Sensors: ");
  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    commSetup[i] = commMuxSetConfig(Wire, SPI, i, commSetup[i]);
    bme[i].begin(BME68X_SPI_INTF, commMuxRead, commMuxWrite, commMuxDelay, &commSetup[i]);

    if (!bme[i].checkStatus()) {
      bme[i].setTPH();
      uint16_t sharedHeatrDur = MEAS_DUR - (bme[i].getMeasDur(BME68X_PARALLEL_MODE) / INT64_C(1000));
      bme[i].setHeaterProf(tempProf, mulProf, sharedHeatrDur, 10);
      bme[i].setOpMode(BME68X_PARALLEL_MODE);
      Serial.print(i);
      Serial.print(" ");
    } else {
      Serial.print("X ");
    }
  }
  Serial.println();
}
