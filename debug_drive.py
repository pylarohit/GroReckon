import numpy as np
import pandas as pd
from georeckon_pipeline import Config, InvariantEKF, phase1_signal_cleaning

def make_realistic_demo(duration_seconds=120):
    np.random.seed(42)
    n_samples = duration_seconds * Config.SAMPLING_RATE_HZ
    dt = Config.DT
    t = np.arange(n_samples) * dt

    speed_kmh = np.zeros(n_samples)
    
    # 0-5s: Parked / Initializing / Calibrating
    # 5-10s: Accelerate from 0 to 60 km/h
    accel_idx_s = int(5.0 / dt)
    accel_idx_e = int(10.0 / dt)
    speed_kmh[accel_idx_s:accel_idx_e] = np.linspace(0.0, 60.0, accel_idx_e - accel_idx_s)
    speed_kmh[accel_idx_e:] = 60.0

    speed_ms = speed_kmh / 3.6
    heading_rad = np.zeros(n_samples)

    # Tunnel blackout from 40s to 80s (40s blackout)
    outage_s = int(40.0 / dt)
    outage_e = int(80.0 / dt)

    true_x = np.zeros(n_samples)
    true_y = np.zeros(n_samples)
    for i in range(1, n_samples):
        true_x[i] = true_x[i-1] + speed_ms[i] * np.cos(heading_rad[i]) * dt
        true_y[i] = true_y[i-1] + speed_ms[i] * np.sin(heading_rad[i]) * dt

    true_accel_forward = np.gradient(speed_ms, dt)
    true_accel_lateral = np.zeros(n_samples)

    # Phone is aligned with vehicle (X forward, Y lateral, Z up)
    acc_x = true_accel_forward + np.random.normal(0, 0.08, n_samples)
    acc_y = true_accel_lateral + np.random.normal(0, 0.08, n_samples)
    acc_z = Config.GRAVITY + np.random.normal(0, 0.08, n_samples)

    gyro_x = np.random.normal(0, 0.001, n_samples)
    gyro_y = np.random.normal(0, 0.001, n_samples)
    gyro_z = np.random.normal(0, 0.001, n_samples)

    lat_origin = 28.6139
    lon_origin = 77.2090
    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.radians(lat_origin))

    gps_lat = lat_origin + true_y / m_lat + np.random.normal(0, 1.0 / m_lat, n_samples)
    gps_lon = lon_origin + true_x / m_lon + np.random.normal(0, 1.0 / m_lon, n_samples)

    # Blackout
    gps_lat[outage_s:outage_e] = np.nan
    gps_lon[outage_s:outage_e] = np.nan

    return pd.DataFrame({
        'Time': t,
        'AccX': acc_x, 'AccY': acc_y, 'AccZ': acc_z,
        'GyroX': gyro_x, 'GyroY': gyro_y, 'GyroZ': gyro_z,
        'GPS_Lat': gps_lat, 'GPS_Lon': gps_lon,
        'GPS_Speed': speed_kmh,
        'GPS_Heading': np.degrees(heading_rad),
        'True_X': true_x, 'True_Y': true_y,
        'True_Speed_ms': speed_ms,
        'True_Heading_rad': heading_rad
    }), outage_s, outage_e

df, out_s, out_e = make_realistic_demo(100)
df_clean = phase1_signal_cleaning(df)

# Phone orientation from parked state (0 to 4s)
first_static = (0, 40)
accel_static = df_clean[['AccX', 'AccY', 'AccZ']].values[first_static[0]:first_static[1]]
f_mean = np.mean(accel_static, axis=0)
pitch = np.arctan2(-f_mean[0], np.sqrt(f_mean[1]**2 + f_mean[2]**2))
roll = np.arctan2(f_mean[1], f_mean[2])

from scipy.spatial.transform import Rotation
R_level = Rotation.from_euler('yx', [pitch, roll]).as_matrix()

# First motion from 5s to 9s (indices 50 to 90)
first_move_accel = df_clean[['AccX', 'AccY', 'AccZ']].values[50:90]
accel_leveled = (R_level @ first_move_accel.T).T
yaw = np.arctan2(np.mean(accel_leveled[:, 1]), np.mean(accel_leveled[:, 0]))
R_yaw = Rotation.from_euler('z', yaw).as_matrix()
R_p2v = R_yaw @ R_level

print("Calibrated R_p2v:\n", np.round(R_p2v, 3))
print("Yaw degrees:", np.degrees(yaw))
