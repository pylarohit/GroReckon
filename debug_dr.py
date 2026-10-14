import json
import numpy as np

# Let's test pure DR propagation with KinoNet forward speed + NHC
from georeckon_pipeline import Config

with open("app/src/main/assets/demo/demo_trip.json") as f:
    d = json.load(f)

samples = d["samples"]
meta = d["metadata"]
out_s = int(meta["outageStartSec"] * 10)
out_e = int(meta["outageEndSec"] * 10)

p = np.array([samples[out_s]["estX"], samples[out_s]["estY"]])
true_p0 = np.array([samples[out_s]["trueLat"], samples[out_s]["trueLon"]])

for i in range(out_s, out_e):
    s = samples[i]
    heading = np.radians(s["headingDeg"])
    speed = s["aiSpeedMps"]
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    p[0] += vx * 0.1
    p[1] += vy * 0.1

s_end = samples[out_e - 1]
true_x = (s_end["trueLon"] - meta["originLon"]) * 111320.0 * np.cos(np.radians(meta["originLat"]))
true_y = (s_end["trueLat"] - meta["originLat"]) * 111320.0

err = np.sqrt((p[0] - true_x)**2 + (p[1] - true_y)**2)
dist = meta["outageDistanceM"]
drift = (err / dist) * 100.0

print(f"Outage Distance: {dist:.1f} m")
print(f"End Error with KinoNet-R2: {err:.2f} m")
print(f"Drift Percent: {drift:.2f}% (Target: < 10.0%)")
