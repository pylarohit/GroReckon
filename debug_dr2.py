import json
import numpy as np

with open("app/src/main/assets/demo/demo_trip.json") as f:
    d = json.load(f)

s_start = d["samples"][350]
s_end = d["samples"][749]

m_lon = 111320.0 * np.cos(np.radians(d["metadata"]["originLat"]))
m_lat = 111320.0

true_x_start = (s_start["trueLon"] - d["metadata"]["originLon"]) * m_lon
true_y_start = (s_start["trueLat"] - d["metadata"]["originLat"]) * m_lat
true_x_end = (s_end["trueLon"] - d["metadata"]["originLon"]) * m_lon
true_y_end = (s_end["trueLat"] - d["metadata"]["originLat"]) * m_lat

true_dx = true_x_end - true_x_start
true_dy = true_y_end - true_y_start
true_dist = np.sqrt(true_dx**2 + true_dy**2)

# Now integrate using AI speeds from start to end
sim_x = true_x_start
sim_y = true_y_start
for i in range(350, 750):
    s = d["samples"][i]
    h = np.radians(s["headingDeg"])
    sp = s["aiSpeedMps"]
    sim_x += sp * np.cos(h) * 0.1
    sim_y += sp * np.sin(h) * 0.1

err = np.sqrt((sim_x - true_x_end)**2 + (sim_y - true_y_end)**2)
drift = (err / true_dist) * 100.0

print(f"True Distance Travelled in Outage: {true_dist:.1f} m")
print(f"Dead Reckoned Endpoint Error:     {err:.2f} m")
print(f"Drift Percentage:                 {drift:.2f}%")
