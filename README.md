# Cooking Activity Detection using IoT Sensors and Machine Learning

## Project Overview
This project aims to detect cooking activities in a household kitchen environment using low-cost IoT sensors and a lightweight Machine Learning model deployed on an ESP32 microcontroller. The system analyzes environmental data such as temperature, humidity, and gas concentrations to distinguish cooking events from normal background conditions.

## Repository Structure
- `data/`: Contains the public dataset and real-world collected logs.
- `models/`: Includes the trained `.pkl` models, scaler, and the generated `model.h` for ESP32.
- `scripts/`: Python scripts for training, validation, and parameter extraction.
- `firmware/`: Arduino/C++ code for data logging and standalone deployment.

## Setup Instructions
1. **Python Environment**:
   - Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib micromlgen`
2. **Hardware**:
   - Connect DHT22, BME680, and MQ-9 sensors to the ESP32.
3. **Training**:
   - Run `scripts/train_model.py` to generate the model artifacts.
4. **Deployment**:
   - Copy `model.h` to the `firmware/esp32_deploy/` folder.
   - Upload the sketch using Arduino IDE.

## Key Results
- **Model**: Random Forest Classifier
- **Target Metrics Achieved**:
  - F1-Score: > 0.70
  - Precision: > 0.80
  - Recall: > 0.60
