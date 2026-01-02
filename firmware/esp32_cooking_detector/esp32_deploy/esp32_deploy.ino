#include "model.h"
#include <DHT.h>
#include <Adafruit_BME680.h>

// Update these with values from export_params.py
float mean[] = {25.1, 44.5, 115.2, 512.0, 400.0, 2890.0};
float std_dev[] = {2.3, 5.1, 35.0, 120.0, 10.0, 600.0};

Eloquent::ML::Port::RandomForest model;
DHT dht(4, DHT22);
Adafruit_BME680 bme;

void setup() {
  Serial.begin(115200);
  dht.begin();
  bme.begin();
}

void loop() {
  if (!bme.performReading()) return;
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  float v = bme.gas_resistance / 1000.0;
  float c = analogRead(34);
  float co2 = 400.0;
  float interaction = t * v;

  float raw[] = {t, h, v, c, co2, interaction};
  float scaled[6];

  for (int i = 0; i < 6; i++) {
    scaled[i] = (raw[i] - mean[i]) / std_dev[i];
  }

  if (model.predict(scaled) == 1) {
    Serial.println("COOKING DETECTED");
  } else {
    Serial.println("NORMAL");
  }
  delay(10000);
}
