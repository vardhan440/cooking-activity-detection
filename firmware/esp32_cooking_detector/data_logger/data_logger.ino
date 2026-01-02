#include <DHT.h>
#include <Adafruit_BME680.h>

DHT dht(4, DHT22);
Adafruit_BME680 bme;

void setup() {
  Serial.begin(115200);
  dht.begin();
  bme.begin();
  Serial.println("Temperature,Relative Humidity,TVOC,CO,CO2,Label");
}

void loop() {
  if (!bme.performReading()) return;
  Serial.print(dht.readTemperature()); Serial.print(",");
  Serial.print(dht.readHumidity()); Serial.print(",");
  Serial.print(bme.gas_resistance / 1000.0); Serial.print(",");
  Serial.print(analogRead(34)); Serial.print(",");
  Serial.print(400); Serial.print(","); // eCO2 placeholder
  Serial.println(0); // Manually edit this to 1 in CSV when cooking
  delay(60000);
}
