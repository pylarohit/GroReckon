import numpy as np
import pandas as pd
from georeckon_pipeline import Config, InvariantEKF, phase1_signal_cleaning, phase2_alignment_and_calibration

# Test with straight acceleration after stop
def make_clean_drive(duration_seconds=60):
    np.random.seed(42)
    n_samples = duration_seconds * Config.SAMPLING_RATE_HZ
    dt = Config.DT
    t = np.arange(n_samples) * dt

    speed_kmh = np.ones(n_samples) * 60.0
    stop_start = int(10 / dt)
    stop_end = int(15 / dt)
    speed_kmh[stop_start:stop_end] = 0.0
    for i in range(stop_start - 15, stop_start):
        speed_kmh[i] = 60.0 * (stop_start - i) / 15
    for i in range(stop_end, min(stop_end + 15, n_samples)):
        speed_kmh[i] = 60.0 * (i - stop_end) / 15

    speed_ms = speed_kmh / 3.6
    heading_rad = np.zeros(n_samples) # Drive straight initially

    # Turn only happens well after acceleration: at 25s
    yaw_rate = np.zeros(n_samples)
    t_turn_s = int(25 / dt)
    t_turn_e = int(27 / dt)
    yaw_rate[t_turn_s:t_turn_e] = 0.08
    for i in range(1, n_samples):
        heading_rad[i] = heading_rad[i-1] + yaw_rate[i] * dt

    true_x = np.zeros(n_samples)
    true_y = np.zeros(n_samples)
    for i in range(1, n_samples):
        true_x[i] = true_x[i-1] + speed_ms[i] * np.cos(heading_rad[i]) * dt
        true_y[i] = true_y[i-1] + speed_ms[i] * np.sin(heading_rad[i]) * dt

    true_accel_forward = np.gradient(speed_ms, dt)
    true_accel_lateral = speed_ms * yaw_rate

    acc_x = true_accel_forward + np.random.normal(0, 0.1, n_samples)
    acc_y = true_accel_lateral + np.random.normal(0, 0.1, n_samples)
    acc_z = Config.GRAVITY + np.random.normal(0, 0.1, n_samples)

    gyro_x = np.random.normal(0, 0.001, n_samples)
    gyro_y = np.random.normal(0, 0.001, n_samples)
    gyro_z = yaw_rate + np.random.normal(0, 0.002, n_samples)

    lat_origin = 28.6139
    lon_origin = 77.2090
    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.radians(lat_origin))
    gps_lat = lat_origin + true_y / m_lat
    gps_lon = lon_origin + true_x / m_lon

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
    })

df = make_clean_drive(50)
df_clean = phase1_signal_cleaning(df)
R_p2v, accel_bias, gyro_bias = phase2_alignment_and_calibration(df_clean)

print("Alignment Yaw matrix:\n", R_p2v)
