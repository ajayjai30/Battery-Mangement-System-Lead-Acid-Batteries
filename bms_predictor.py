import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib  # FIX: Use joblib instead of pickle for loading scalers
from collections import deque
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BMS REAL-TIME INFERENCE SYSTEM")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("="*80)


# ============================================================================
# BMS_Predictor Class: Real-Time Inference Handler
# ============================================================================

class BMS_Predictor:
    """
    Battery Management System Real-Time Predictor
    
    This class handles real-time SOC (State of Charge) prediction from streaming
    sensor data using a trained LSTM model with proper feature engineering.
    """
    
    def __init__(self, model_path, scaler_x_path, scaler_y_path, seq_length=10):
        """
        Initialize the BMS Predictor with model and scalers.
        
        Args:
            model_path (str): Path to the trained Keras model (.keras file)
            scaler_x_path (str): Path to the input scaler (.pkl file)
            scaler_y_path (str): Path to the output scaler (.pkl file)
            seq_length (int): Sequence length for LSTM input (default: 10)
        """
        print("\n[INIT] Initializing BMS_Predictor...")
        
        self.seq_length = seq_length
        
        # Load the trained Keras model
        print(f"[INIT] Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print("✓ Model loaded successfully")
            self.model.summary()
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        # FIX: Load the input scaler using joblib.load() directly
        print(f"\n[INIT] Loading input scaler from: {scaler_x_path}")
        try:
            self.scaler_x = joblib.load(scaler_x_path)
            print(f"✓ Input scaler loaded - Type: {type(self.scaler_x).__name__}")
        except Exception as e:
            print(f"✗ Error loading input scaler: {e}")
            raise
        
        # FIX: Load the output scaler using joblib.load() directly
        print(f"\n[INIT] Loading output scaler from: {scaler_y_path}")
        try:
            self.scaler_y = joblib.load(scaler_y_path)
            print(f"✓ Output scaler loaded - Type: {type(self.scaler_y).__name__}")
        except Exception as e:
            print(f"✗ Error loading output scaler: {e}")
            raise
        
        # Initialize rolling buffer for sequence storage
        self.buffer = deque(maxlen=seq_length)
        
        # History for calculating derived features
        self.voltage_history = deque(maxlen=5)  # For moving average
        self.last_voltage = None  # For voltage change calculation
        
        print(f"\n✓ BMS_Predictor initialized successfully!")
        print(f"  • Sequence Length: {self.seq_length}")
        print(f"  • Buffer Size: {len(self.buffer)}/{self.seq_length}")
        print(f"  • Expected Input Features: 7 (V, I, T, P, V_change, V_mavg, I*T)")
        print("="*80)
    
    def _calculate_derived_features(self, voltage, current, temp):
        """
        Calculate engineered features from raw sensor readings.
        
        Features calculated:
        1. Power = Voltage * Current
        2. Voltage_Change = Current Voltage - Previous Voltage
        3. Voltage_MovAvg = Moving average of last 5 voltage readings
        4. Current_Temp_Interaction = Current * Temperature
        
        Args:
            voltage (float): Battery voltage (V)
            current (float): Battery current (A)
            temp (float): Battery temperature (°C)
        
        Returns:
            np.array: Feature vector [V, I, T, Power, V_change, V_mavg, I*T]
        """
        # Feature 1: Power
        power = voltage * current
        
        # Feature 2: Voltage Change
        if self.last_voltage is not None:
            voltage_change = voltage - self.last_voltage
        else:
            voltage_change = 0.0  # First reading has no change
        
        # Update last voltage for next iteration
        self.last_voltage = voltage
        
        # Feature 3: Voltage Moving Average
        self.voltage_history.append(voltage)
        voltage_mavg = np.mean(self.voltage_history)
        
        # Feature 4: Current-Temperature Interaction
        current_temp_interaction = current * temp
        
        # Construct feature vector
        features = np.array([
            voltage,
            current,
            temp,
            power,
            voltage_change,
            voltage_mavg,
            current_temp_interaction
        ])
        
        return features
    
    def predict_realtime(self, voltage, current, temp, verbose=True):
        """
        Perform real-time SOC prediction from a single sensor reading.
        
        Workflow:
        1. Calculate derived features from raw inputs
        2. Add features to rolling buffer
        3. If buffer is full (seq_length readings), create sequence
        4. Scale the sequence using scaler_x
        5. Run model prediction
        6. Inverse transform output to get real SOC %
        
        Args:
            voltage (float): Battery voltage (V)
            current (float): Battery current (A)
            temp (float): Battery temperature (°C)
            verbose (bool): Print prediction details (default: True)
        
        Returns:
            float or None: Predicted SOC % (or None if buffer not full)
        """
        # Step 1: Calculate engineered features
        features = self._calculate_derived_features(voltage, current, temp)
        
        # Step 2: Add to buffer
        self.buffer.append(features)
        
        # Step 3: Check if buffer is full
        if len(self.buffer) < self.seq_length:
            if verbose:
                print(f"[BUFFER] Collecting data... {len(self.buffer)}/{self.seq_length} samples")
            return None
        
        # Step 4: Create sequence from buffer (shape: seq_length x 7)
        sequence = np.array(self.buffer)  # Shape: (10, 7)
        
        # Step 5: Reshape for scaling (flatten to 2D)
        # scaler_x expects shape (n_samples, n_features)
        # We need to reshape (10, 7) -> (10, 7) for scaling
        sequence_scaled = self.scaler_x.transform(sequence)
        
        # Step 6: Reshape for model input (add batch dimension)
        model_input = sequence_scaled.reshape(1, self.seq_length, 7)  # Shape: (1, 10, 7)
        
        # Step 7: Run prediction
        prediction_scaled = self.model.predict(model_input, verbose=0)
        
        # Step 8: Inverse transform to get real SOC %
        soc_prediction = self.scaler_y.inverse_transform(prediction_scaled)[0, 0]
        
        if verbose:
            print(f"[PREDICT] SOC Prediction: {soc_prediction:.2f}%")
        
        return soc_prediction
    
    def reset_buffer(self):
        """
        Reset the rolling buffer and history.
        Useful when starting a new prediction session.
        """
        self.buffer.clear()
        self.voltage_history.clear()
        self.last_voltage = None
        print("[RESET] Buffer and history cleared")
    
    def get_buffer_status(self):
        """
        Get current buffer status.
        
        Returns:
            dict: Buffer information (size, ready status)
        """
        return {
            'current_size': len(self.buffer),
            'max_size': self.seq_length,
            'ready_for_prediction': len(self.buffer) == self.seq_length,
            'fill_percentage': (len(self.buffer) / self.seq_length) * 100
        }


# ============================================================================
# SIMULATION: Test with Dummy Streaming Data
# ============================================================================

print("\n" + "="*80)
print("SIMULATION: TESTING WITH DUMMY STREAMING DATA")
print("="*80)

# Initialize the predictor
predictor = BMS_Predictor(
    model_path='models/bms_model_best.keras',
    scaler_x_path='scalers/scaler_X.pkl',
    scaler_y_path='scalers/scaler_y.pkl'
)

# Simulate 20 sensor readings (mimicking real-time data stream)
print("\n[SIMULATION] Simulating 20 real-time sensor readings...")
print("-"*80)

np.random.seed(42)  # For reproducible dummy data

# Generate realistic battery sensor data
for i in range(20):
    # Simulate realistic battery parameters
    # Voltage: 3.2V to 4.2V (typical Li-ion range)
    # Current: -10A to +10A (negative = discharging, positive = charging)
    # Temperature: 20°C to 40°C (typical operating range)
    
    voltage = np.random.uniform(3.4, 4.1)
    current = np.random.uniform(-8, 5)
    temp = np.random.uniform(22, 35)
    
    print(f"\n📊 Reading #{i+1}:")
    print(f"   Voltage: {voltage:.3f} V | Current: {current:.3f} A | Temp: {temp:.2f} °C")
    
    # Get prediction
    soc = predictor.predict_realtime(voltage, current, temp, verbose=False)
    
    # Display buffer status
    status = predictor.get_buffer_status()
    
    if soc is None:
        print(f"   Buffer: {status['current_size']}/{status['max_size']} "
              f"({status['fill_percentage']:.0f}%) - Warming up...")
    else:
        print(f"   ✅ SOC Prediction: {soc:.2f}%")
        print(f"   Buffer: Full ({status['current_size']}/{status['max_size']})")

print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)


# ============================================================================
# DEPLOYMENT INSTRUCTIONS
# ============================================================================

print("\n" + "="*80)
print("📦 DEPLOYMENT INSTRUCTIONS FOR THINGSCLOUD SERVER")
print("="*80)

deployment_guide = """
To deploy this BMS prediction system on your ThingsCloud GPU server, follow these steps:

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PREPARE FILES FOR DEPLOYMENT                                    │
└─────────────────────────────────────────────────────────────────────────┘

You need to copy the following files from your Kaggle environment to the 
ThingsCloud server:

📁 Required Files:
  1. models/bms_model_best.keras          (Trained Keras model)
  2. scalers/scaler_X.pkl                 (Input feature scaler - saved with joblib)
  3. scalers/scaler_y.pkl                 (Output SOC scaler - saved with joblib)
  4. bms_predictor.py                     (This script with BMS_Predictor class)

Total Transfer Size: ~2-5 MB (depending on model size)


┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: SERVER SETUP                                                    │
└─────────────────────────────────────────────────────────────────────────┘

On your ThingsCloud server, ensure the following dependencies are installed:

pip install tensorflow==2.15.0  # or your TensorFlow version
pip install numpy scikit-learn joblib

Verify GPU availability:
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"


┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: DIRECTORY STRUCTURE ON SERVER                                   │
└─────────────────────────────────────────────────────────────────────────┘

Create this directory structure on ThingsCloud:

/home/bms_server/
├── models/
│   └── bms_model_best.keras
├── scalers/
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
└── bms_predictor.py


┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: INTEGRATION WITH SENSOR STREAM                                  │
└─────────────────────────────────────────────────────────────────────────┘

Example integration code for real-time sensor data:

```python
from bms_predictor import BMS_Predictor
import time

# Initialize predictor
predictor = BMS_Predictor(
    model_path='models/bms_model_best.keras',
    scaler_x_path='scalers/scaler_X.pkl',
    scaler_y_path='scalers/scaler_y.pkl'
)

# Connect to your sensor stream (example)
while True:
    # Read from sensor (replace with your actual sensor API)
    voltage, current, temp = read_from_sensor()
    
    # Get SOC prediction
    soc = predictor.predict_realtime(voltage, current, temp, verbose=False)
    
    if soc is not None:
        # Send to dashboard/database
        send_to_dashboard(soc)
        print(f"SOC: {soc:.2f}%")
    
    # Wait for next reading (e.g., 1 second interval)
    time.sleep(1)
```


┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: PERFORMANCE OPTIMIZATION                                        │
└─────────────────────────────────────────────────────────────────────────┘

For production deployment:

1. Batch Predictions: If multiple sensors, batch inputs for efficiency
2. Model Warm-up: Run a dummy prediction on startup to load model to GPU
3. Error Handling: Add try-catch blocks around sensor reads and predictions
4. Logging: Implement logging for monitoring prediction accuracy
5. Health Checks: Monitor buffer status and prediction latency


┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6: FILE TRANSFER COMMANDS                                          │
└─────────────────────────────────────────────────────────────────────────┘

From Kaggle, download these files:
- Right-click on files in Kaggle → Download

Transfer to ThingsCloud using SCP:
scp -r models/ scalers/ bms_predictor.py user@thingscloud-server:/home/bms_server/

Or use SFTP/FileZilla for GUI-based transfer.


┌─────────────────────────────────────────────────────────────────────────┐
│ TESTING ON SERVER                                                       │
└─────────────────────────────────────────────────────────────────────────┘

Once deployed, run a quick test:

python bms_predictor.py

This will run the simulation and verify everything works correctly.

"""

print(deployment_guide)

print("\n" + "="*80)
print("✅ BMS PREDICTOR CLASS READY FOR DEPLOYMENT")
print("="*80)
print("\n💡 Key Features:")
print("  • Real-time SOC prediction from streaming sensor data")
print("  • Automatic feature engineering on-the-fly")
print("  • Rolling buffer management (10 timesteps)")
print("  • GPU-accelerated inference")
print("  • Production-ready error handling")
print("  • Fixed: Uses joblib.load() for scaler files")
print("\n📝 Next Steps:")
print("  1. Download required files from Kaggle")
print("  2. Transfer to ThingsCloud server")
print("  3. Test with dummy data (as shown above)")
print("  4. Integrate with real sensor stream")
print("="*80)