import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

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
        self.seq_length = seq_length
        
        # Load the trained Keras model
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")
        
        # Load the input scaler
        try:
            self.scaler_x = joblib.load(scaler_x_path)
        except Exception as e:
            raise RuntimeError(f"Error loading input scaler from {scaler_x_path}: {e}")
        
        # Load the output scaler
        try:
            self.scaler_y = joblib.load(scaler_y_path)
        except Exception as e:
            raise RuntimeError(f"Error loading output scaler from {scaler_y_path}: {e}")
        
        # Initialize rolling buffer for sequence storage
        self.buffer = deque(maxlen=seq_length)
        
        # History for calculating derived features
        self.voltage_history = deque(maxlen=5)  # For moving average
        self.last_voltage = None  # For voltage change calculation
    
    def _calculate_derived_features(self, voltage, current, temp):
        """
        Calculate engineered features from raw sensor readings.
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
        Returns:
            float or None: Predicted SOC % (or None if buffer not full)
        """
        # Step 1: Calculate engineered features
        features = self._calculate_derived_features(voltage, current, temp)
        
        # Step 2: Add to buffer
        self.buffer.append(features)
        
        # Step 3: Check if buffer is full
        if len(self.buffer) < self.seq_length:
            return None
        
        # Step 4: Create sequence from buffer (shape: seq_length x 7)
        sequence = np.array(self.buffer)
        
        # Step 5: Reshape for scaling (flatten to 2D)
        sequence_scaled = self.scaler_x.transform(sequence)
        
        # Step 6: Reshape for model input (add batch dimension)
        model_input = sequence_scaled.reshape(1, self.seq_length, 7)
        
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
        """
        self.buffer.clear()
        self.voltage_history.clear()
        self.last_voltage = None
    
    def get_buffer_status(self):
        """
        Get current buffer status.
        """
        return {
            'current_size': len(self.buffer),
            'max_size': self.seq_length,
            'ready_for_prediction': len(self.buffer) == self.seq_length,
            'fill_percentage': (len(self.buffer) / self.seq_length) * 100
        }

# ============================================================================
# MAIN GUARD: Only runs if file is executed directly (not when imported)
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("BMS PREDICTOR LIBRARY")
    print("Run this file purely to test class instantiation.")
    print("="*80)
    
    try:
        # Example test usage
        print("Testing initialization...")
        # Note: Paths here assume default directory structure. 
        # Update them if running from a different location.
        predictor = BMS_Predictor(
            model_path='models/bms_model_best.keras',
            scaler_x_path='scalers/scaler_X.pkl',
            scaler_y_path='scalers/scaler_y.pkl'
        )
        print("✅ BMS_Predictor initialized successfully.")
    except Exception as e:
        print(f"⚠️ Initialization skipped (Files likely missing): {e}")
