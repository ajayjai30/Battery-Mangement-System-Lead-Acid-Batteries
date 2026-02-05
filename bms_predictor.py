import numpy as np
import tensorflow as tf
import joblib
import os
from collections import deque
from datetime import datetime

class BMSPredictor:
    """
    SOH (State of Health) Predictor for Lead-Acid Batteries.
    Uses an LSTM model with Robust Smoothing to track degradation.
    """
    
    def __init__(self, model_dir='models', scaler_dir='scalers'):
        print("❤️ Initializing SOH Prediction Engine...")
        
        # 1. Load Model
        # We prioritize the GPU/Keras model for Python environments
        self.model = None
        try:
            model_path = os.path.join(model_dir, 'soh_model_gpu.keras')
            # Fallback if specific gpu name not found
            if not os.path.exists(model_path):
                 model_path = os.path.join(model_dir, 'soh_model_robust.keras')
            
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"✓ Loaded SOH Model: {model_path}")
            else:
                print(f"❌ Error: Model file not found in {model_dir}")
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")

        # 2. Load Scalers
        try:
            self.scaler_X = joblib.load(os.path.join(scaler_dir, 'scaler_X_soh.pkl'))
            self.scaler_y = joblib.load(os.path.join(scaler_dir, 'scaler_y_soh.pkl'))
            print("✓ Loaded SOH Scalers")
        except FileNotFoundError:
            print(f"❌ CRITICAL: Scalers not found in {scaler_dir}! Run training pipeline first.")
            return

        # 3. Memory & Smoothing
        self.seq_length = 100 # Must match the training sequence length
        self.data_buffer = deque(maxlen=self.seq_length) # Stores raw features for LSTM
        self.soh_smoothing = deque(maxlen=20) # Average last 20 predictions for stability
        
        # 4. Feature Engineering Memory (State variables)
        self.prev_v = None
        self.prev_i = None
        self.cumulative_energy = 0.0
        self.last_time = datetime.now()

    def _engineer_features(self, v, i, t):
        """
        Calculates advanced SOH indicators in real-time.
        Order must match training: [V, I, Temp, Res_Proxy, Energy, Stress]
        """
        now = datetime.now()
        # Calculate time difference in hours (for Energy integration)
        dt_seconds = (now - self.last_time).total_seconds()
        dt_hours = dt_seconds / 3600.0
        
        # Avoid huge jumps on first run
        if dt_hours > 1.0: dt_hours = 0.0 
        
        # 1. Internal Resistance Proxy (|dV / dI|)
        # This is the primary physical indicator of SOH
        ir_proxy = 0.0
        if self.prev_v is not None and self.prev_i is not None:
            di = abs(i - self.prev_i)
            dv = abs(v - self.prev_v)
            # Only calculate if current change is significant (to avoid noise division)
            if di > 0.05: 
                ir_proxy = dv / di
        
        # 2. Cumulative Energy (Throughput)
        # SOH degrades as total energy processed increases
        power = abs(v * i)
        self.cumulative_energy += (power * dt_hours)
        
        # 3. Thermal Stress Index
        # High current at high temp accelerates aging
        stress = abs(i) * t
        
        # Update state for next loop
        self.prev_v = v
        self.prev_i = i
        self.last_time = now
        
        return [v, i, t, ir_proxy, self.cumulative_energy, stress]

    def predict_realtime(self, voltage, current, temp):
        """
        Main inference method.
        Args:
            voltage (float): Battery Voltage (V)
            current (float): Battery Current (A) - Negative for discharge
            temp (float): Battery Temperature (C)
        Returns: 
            float: SOH percentage (0.0 to 100.0)
            None: If buffer is still filling
        """
        if self.model is None:
            return None

        # Step 1: Feature Engineering
        features = self._engineer_features(voltage, current, temp)
        self.data_buffer.append(features)
        
        # Wait for buffer to fill (LSTM needs history)
        if len(self.data_buffer) < self.seq_length:
            return None
        
        # Step 2: Prepare Input
        # Convert buffer to numpy array (1, 100, 6)
        input_raw = np.array(self.data_buffer)
        
        # Normalize using the trained scaler
        input_scaled = self.scaler_X.transform(input_raw)
        
        # Reshape for LSTM [Batch, Timesteps, Features]
        input_seq = input_scaled.reshape(1, self.seq_length, -1)
        
        # Step 3: AI Prediction
        # verbose=0 prevents flooding console with progress bars
        raw_soh_scaled = self.model.predict(input_seq, verbose=0)
        
        # Convert back to real percentage
        raw_soh = self.scaler_y.inverse_transform(raw_soh_scaled)[0, 0]
        
        # Step 4: Robust Smoothing
        # Add to smoothing buffer
        self.soh_smoothing.append(raw_soh)
        
        # Calculate average of the buffer
        smooth_soh = sum(self.soh_smoothing) / len(self.soh_smoothing)
        
        # Step 5: Safety Clamping
        # SOH cannot physically be < 0% or > 100%
        final_soh = max(0.0, min(100.0, smooth_soh))
        
        return final_soh

    def reset_history(self):
        """Resets the history buffer (useful for testing)"""
        self.data_buffer.clear()
        self.soh_smoothing.clear()
        self.cumulative_energy = 0.0
        print("Dataset history reset.")
