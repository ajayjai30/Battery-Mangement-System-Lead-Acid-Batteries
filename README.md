# 🔋 AI Battery State of Health (SOH) Predictor

---

## 📌 Project Overview

This project implements an **Artificial Intelligence–based Battery Management System (BMS)** explicitly designed to predict the **State of Health (SOH)** of batteries.

While **State of Charge (SOC)** acts as a *"fuel gauge"* indicating how much energy remains for immediate use, **SOH** acts as a *"physician's report,"* revealing the overall lifespan and structural integrity of the battery. It is crucial for detecting long-term degradation, internal resistance buildup, and capacity fade before a catastrophic failure occurs.

The system utilizes a **Deep LSTM (Long Short-Term Memory)** neural network, trained on high-fidelity aging profiles from the **NASA Prognostics Center of Excellence (PCoE)**. Unlike simple lookup tables that fail as batteries age, this model analyzes the *"slope"* and velocity of voltage and temperature changes over a **10-step time window**. By observing these temporal dynamics, the AI can estimate the battery's remaining useful capacity with high accuracy, even under varying load conditions.

---

## 📊 Performance Metrics

The model has been rigorously trained and optimized for **low-latency inference**, making it suitable for deployment on **edge devices** (like Raspberry Pi or ESP32) or **cloud-based monitoring servers**.

| **Metric** | **Value** | **Description** |
|------------|-----------|-----------------|
| **Model Architecture** | Deep LSTM | 2 Stacked Layers (128 & 64 Units) with Dropout for robust generalization. |
| **Input Window** | 10 Steps | Captures the last ~2.5 minutes of sensor history to determine trend direction. |
| **MAE (Error)** | ~4.23% | The average deviation from the actual health. (e.g., A prediction of 80% is reliably between 76–84%). |
| **RMSE** | ~5.86% | Root Mean Square Error, indicating a low susceptibility to extreme outlier errors. |
| **Inference Time** | < 50ms | Highly efficient, enabling real-time health monitoring without system lag. |

---

## 🧠 How It Works

The system operates on a continuous feedback loop, transforming raw, noisy sensor data into actionable health insights:

### 1. Data Ingestion
The system pulls live sensor streams ($Voltage$, $Current$, $Temperature$) via the ThingSpeak API or a direct local serial connection. This stage handles data synchronization to ensure valid inputs.

### 2. Windowing (Temporal Context)
Instead of analyzing a single data point, the system buffers the last **10 data points**.

**The "Why":**  
A degraded battery and a healthy battery might show the same voltage at rest. However, under load, a degraded battery's voltage drops significantly faster. The LSTM requires this *"moving window"* to detect that specific decay curve.

### 3. Inference
The buffered data is normalized (scaled to 0–1 range) to match the training distribution and fed into the `.keras` model. The LSTM layers process the sequence to extract deep temporal features related to chemical aging.

### 4. Output
The model outputs a single scalar value ($0\% - 100\%$) representing the SOH. This is calculated as the ratio of the Current Maximum Capacity to the Original Nominal Capacity.

---

## 📂 Project Structure

```bash
├── Model/
│   ├── soh_model_gpu.keras            # GPU-optimized Keras model
│   └── soh_model_robust.tflite        # TFLite robustness model
├── Scalers/
│   ├── scaler_X_soh.pkl               # Pre-fitted scaler for normalizing input sensors (V, I, T)
│   └── scaler_y_soh.pkl               # Decodes the model's output back to a readable SOH %
├── bms_predictor.py                   # Core AI Class: Manages the rolling buffer, feature extraction, and inference
├── FINAL_IMPLEMENTATION_BMS_(SOH_PREDICTION).ipynb # Full implementation Jupyter Notebook with Gradio UI
├── training-pipeline-for-bms-soh-prediction.ipynb  # The Jupyter Notebook used to train the model
├── processed_bms_data.zip             # Processed dataset file
├── requirements.txt                   # List of necessary Python dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ajayjai30/Battery-Mangement-System-Lead-Acid-Batteries.git
cd Battery-Mangement-System-Lead-Acid-Batteries
```

---

### 2. Install Dependencies

Ensure you are running a compatible Python environment (**3.10+**) and install the required libraries:

```bash
pip install -r requirements.txt
```

---

### 3. Usage (Real-Time Cloud Loop via Gradio)

To start the automated bridge that pulls live data from ThingSpeak, processes the health metrics, and logs the SOH to the console, you can use the interactive Gradio App inside the notebook:

1. Open `FINAL_IMPLEMENTATION_BMS_(SOH_PREDICTION).ipynb`.
2. Run all cells to launch the Gradio Dashboard.
3. Select "Live ThingSpeak" mode and enter your `THINGSPEAK_CHANNEL_ID` and `API_KEY` to stream data.

---

### 4. Usage (Custom Python Script)

You can easily import the predictor logic into your own custom dashboard or automation script:

```python
from bms_predictor import BMSPredictor

# Initialize the AI engine with the trained artifacts
ai = BMSPredictor(
    model_dir='Model',
    scaler_dir='Scalers'
)

# Simulate feeding live data (Voltage, Current, Temp)
# The predictor returns 'None' while it fills the 10-step buffer
soh_prediction = ai.predict_realtime(12.4, -1.5, 28.5)

if soh_prediction:
    print(f"Current Battery Health: {soh_prediction:.2f}%")
else:
    print("Buffering data... (Need 10 data points)")
```
