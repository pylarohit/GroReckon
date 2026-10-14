import numpy as np
import pandas as pd
from georeckon_pipeline import Config, _generate_synthetic_drive_data, phase1_signal_cleaning

df = _generate_synthetic_drive_data(50)
df_clean = phase1_signal_cleaning(df)
accel_data = df_clean[['AccX', 'AccY', 'AccZ']].values

# first static segment
# In synthetic drive, stop is at 200:250 (20s-25s)
print("Static 20-25s mean AccX:", np.mean(accel_data[200:250, 0]))
print("Motion 25-27s mean AccX:", np.mean(accel_data[250:270, 0]))
print("Motion 25-27s mean AccY:", np.mean(accel_data[250:270, 1]))
print("Motion 25-30s mean AccX:", np.mean(accel_data[250:300, 0]))
print("Motion 25-30s mean AccY:", np.mean(accel_data[250:300, 1]))
