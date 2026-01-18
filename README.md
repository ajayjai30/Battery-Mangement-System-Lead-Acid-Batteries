# Battery-Mangement-System - Lead-Acid-Batteries


================================================================================
AI BATTERY MANAGEMENT SYSTEM (BMS) WITH THINGSPEAK INTEGRATION
================================================================================
Project: AI-Based SOC Prediction Engine (LSTM)
Author: [Your Name]
Date: January 2026

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements an Artificial Intelligence-based Battery Management System
(BMS) designed to predict the State of Charge (SOC) of batteries in real-time.

Unlike traditional voltage-lookup methods which are inaccurate under load, this
system uses a Long Short-Term Memory (LSTM) neural network. It analyzes the
history (temporal sequences) of Voltage, Current, and Temperature to predict
battery life with high accuracy (>99% correlation).

The system is designed for a hybrid architecture:
1. Sensors push data to the ThingSpeak Cloud.
2. This AI Engine fetches the data from the cloud.
3. The AI predicts the SOC and logs it back or alerts the user.

--------------------------------------------------------------------------------
2. DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
Ensure your project folder is organized exactly as follows:

/your_project_root/
├── bms_predictor.py           # The Main Inference Engine Class
├── thingspeak_bridge.py       # Script to fetch data from Cloud & Predict
├── models/
│   └── bms_model_best.keras   # The trained LSTM Model (The "Brain")
├── scalers/
│   ├── scaler_X.pkl           # Input Feature Scaler
│   └── scaler_y.pkl           # Target SOC Scaler
└── README.txt                 # This file

--------------------------------------------------------------------------------
3. PREREQUISITES & INSTALLATION
--------------------------------------------------------------------------------
The system requires Python 3.8+ and the following libraries.

Run this command to install dependencies:
    pip install tensorflow numpy scikit-learn joblib pandas requests

*Note:* If running on a GPU-enabled server, ensure CUDA drivers are installed
for faster inference.

--------------------------------------------------------------------------------
4. HOW IT WORKS (THE AI PIPELINE)
--------------------------------------------------------------------------------
A. DATA PROCESSING:
   The raw sensor data (Voltage, Current, Temp) is passed to the 'BMS_Predictor'
   class. It automatically:
   - Buffers the last 10 readings (Sliding Window).
   - Calculates derived physics features (Power = V*I, dV/dt, Moving Averages).
   - Scales the data using the pre-fitted scalers.

B. PREDICTION:
   The processed sequence is fed into the LSTM model.
   - Model Architecture: LSTM(128) -> Dropout -> LSTM(64) -> Dense(1).
   - Output: A precise SOC percentage (0.00% to 100.00%).

--------------------------------------------------------------------------------
5. USAGE GUIDE: RUNNING WITH THINGSPEAK
--------------------------------------------------------------------------------
To connect this AI to your ThingSpeak cloud data, use the 'thingspeak_bridge.py'
script.

A. SETUP THINGSPEAK:
   Ensure your ThingSpeak Channel has these fields:
   - Field 1: Voltage (V)
   - Field 2: Current (A)
   - Field 3: Temperature (°C)

B. CREATE THE BRIDGE SCRIPT:
   Create a file named 'thingspeak_bridge.py' with the following code:

   ```python
   import requests
   import time
   from bms_predictor import BMS_Predictor

   # --- CONFIGURATION ---
   CHANNEL_ID = "YOUR_CHANNEL_ID_HERE"
   READ_API_KEY = "YOUR_READ_API_KEY_HERE"
   # URL to fetch the single latest feed
   THINGSPEAK_URL = f"[https://api.thingspeak.com/channels/](https://api.thingspeak.com/channels/){CHANNEL_ID}/feeds/last.json?api_key={READ_API_KEY}"

   # 1. Initialize the AI Engine
   print("🔋 Initializing AI BMS Engine...")
   ai_engine = BMS_Predictor(
       model_path='models/bms_model_best.keras',
       scaler_x_path='scalers/scaler_X.pkl',
       scaler_y_path='scalers/scaler_y.pkl'
   )

   # 2. Main Execution Loop
   print("✅ AI Engine Ready. Listening to ThingSpeak...")
   while True:
       try:
           # Fetch latest data from Cloud
           response = requests.get(THINGSPEAK_URL)
           data = response.json()

           # Parse the sensor values
           # Note: Ensure your microcontroller sends data to these specific fields
           voltage = float(data['field1'])
           current = float(data['field2'])
           temp = float(data['field3'])

           # Get AI Prediction
           # Returns None if buffer isn't full (first 10 readings)
           soc = ai_engine.predict_realtime(voltage, current, temp)

           if soc is not None:
               print(f"📊 Sensor: {voltage}V, {current}A, {temp}°C")
               print(f"🔋 PREDICTED SOC: {soc:.2f}%")
               print("-" * 30)
           else:
               print("⏳ Buffering data (Need 10 data points)...")

       except Exception as e:
           print(f"❌ Connection Error: {e}")

       # Wait 15 seconds before next poll (ThingSpeak free limit)
       time.sleep(15)
```
C. RUN THE BRIDGE: python thingspeak_bridge.py

TROUBLESHOOTING

"UnpicklingError": Ensure you are using 'joblib.load()' inside bms_predictor.py, not 'pickle.load()'.

"Buffering data..." forever: The model needs 10 consecutive data points to make a prediction. Ensure your loop runs at least 10 times.

GPU Errors: If the server lacks a GPU, TensorFlow usually falls back to CPU automatically. If you see CUDA errors, you can force CPU mode by adding this to the top of your script: import os os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

================================================================================ END OF README
